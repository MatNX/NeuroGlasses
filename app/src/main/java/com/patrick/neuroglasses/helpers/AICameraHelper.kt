package com.patrick.neuroglasses.helpers

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.example.cxrglobal.callbacks.IImageStreamCbk

/**
 * Helper class for AI scene camera operations through the authorized Hi Rokid channel.
 * This handles camera operations that return WebP image data as byte arrays.
 */
class AICameraHelper(private val appTag: String = "AICameraHelper") {

    // Camera state
    var isCameraOpen = false
        private set

    private val handler = Handler(Looper.getMainLooper())
    private var captureInProgress = false
    private var captureGeneration = 0
    private var captureAttempts: List<PhotoRequest> = emptyList()
    private var captureAttemptIndex = 0

    // Callback interface for camera events
    interface AICameraListener {
        fun onCameraOpened(message: String)
        fun onCameraOpenFailed(message: String)
        fun onPhotoStatusUpdate(message: String)
        fun onPhotoSuccess(bitmap: Bitmap, dataSize: Int, width: Int, height: Int)
        fun onPhotoFailed(message: String)
    }

    private var listener: AICameraListener? = null

    fun setListener(listener: AICameraListener) {
        this.listener = listener
        RokidHostConnection.setImageStreamListener(imageStreamCallback)
    }

    /**
     * Image callback with comprehensive logging
     */
    private val imageStreamCallback = object : IImageStreamCbk {
        override fun onImageReceived(data: ByteArray) {
            handlePhotoData(data)
        }

        override fun onImageError(code: Int, msg: String?) {
            Log.e(appTag, "Photo capture failed: $code $msg")
            retryOrFail("Photo failed: ${msg ?: code}")
        }
    }

    private fun handlePhotoData(photo: ByteArray) {
        Log.d(appTag, "=== Photo Result Callback Triggered ===")
        Log.d(appTag, "Photo data size: ${photo.size} bytes")

        if (!captureInProgress) {
            Log.w(appTag, "Ignoring image callback because no photo capture is active")
            return
        }

        if (photo.isEmpty()) {
            Log.w(appTag, "Photo captured but data is empty!")
            retryOrFail("Photo captured but no data received")
            return
        }

        Log.i(appTag, "Photo data received: ${photo.size} bytes (WebP format)")

        // Log first few bytes for debugging (header check)
        if (photo.size >= 12) {
            val headerBytes = photo.take(12).joinToString(" ") {
                "%02X".format(it)
            }
            Log.d(appTag, "Photo header bytes: $headerBytes")
        }

        try {
            Log.d(appTag, "Attempting to decode WebP image data...")
            val bitmap = BitmapFactory.decodeByteArray(photo, 0, photo.size)

            if (bitmap != null) {
                Log.i(appTag, "WebP decoded successfully! Bitmap size: ${bitmap.width}x${bitmap.height}")
                Log.d(appTag, "Bitmap config: ${bitmap.config}")
                Log.d(appTag, "Bitmap byte count: ${bitmap.byteCount}")

                completePhotoCapture()
                listener?.onPhotoSuccess(bitmap, photo.size, bitmap.width, bitmap.height)
            } else {
                Log.e(appTag, "Failed to decode WebP - BitmapFactory returned null")
                retryOrFail("Photo received but decode failed")
            }
        } catch (e: Exception) {
            Log.e(appTag, "Exception while decoding WebP image", e)
            Log.e(appTag, "Exception type: ${e.javaClass.simpleName}")
            Log.e(appTag, "Exception message: ${e.message}")
            retryOrFail("Photo decode error: ${e.message}")
        }

        Log.d(appTag, "=== Photo Result Callback Complete ===")
    }

    /**
     * Open the glass camera for AI scene photo capture
     * This is optional - takePhoto can be called directly
     * @param width photo width (default 1920)
     * @param height photo height (default 1080)
     * @param quality photo quality 0-100 (default 85)
     */
    fun openGlassCamera(
        width: Int = DEFAULT_PHOTO_WIDTH,
        height: Int = DEFAULT_PHOTO_HEIGHT,
        quality: Int = DEFAULT_PHOTO_QUALITY
    ): Boolean {
        Log.d(appTag, "=== Opening Glass Camera ===")
        Log.d(appTag, "Camera parameters - Width: $width, Height: $height, Quality: $quality")

        if (RokidHostConnection.isConnected()) {
            Log.i(appTag, "Camera channel ready through Hi Rokid service")
            isCameraOpen = true
            RokidHostConnection.setImageStreamListener(imageStreamCallback)
            listener?.onCameraOpened("Camera ready (${width}x${height}, Q:$quality)")
        } else {
            Log.e(appTag, "Hi Rokid channel is not connected")
            isCameraOpen = false
            listener?.onCameraOpenFailed("Hi Rokid channel is not connected")
        }

        Log.d(appTag, "=== Open Camera Complete ===")
        return isCameraOpen
    }

    /**
     * Take a photo using the glasses camera in the AI scene.
     * @param width photo width (default 1280)
     * @param height photo height (default 720)
     * @param quality photo quality 0-100 (default 85)
     */
    fun takePhoto(
        width: Int = DEFAULT_PHOTO_WIDTH,
        height: Int = DEFAULT_PHOTO_HEIGHT,
        quality: Int = DEFAULT_PHOTO_QUALITY
    ) {
        Log.d(appTag, "=== Take Photo Requested ===")
        Log.d(appTag, "Photo parameters - Width: $width, Height: $height, Quality: $quality")
        Log.d(appTag, "Camera open: $isCameraOpen")

        if (!openGlassCamera(width, height, quality)) {
            Log.e(appTag, "Photo capture aborted because camera/channel is not ready")
            listener?.onPhotoFailed("Hi Rokid channel is not connected")
            return
        }

        captureGeneration += 1
        captureInProgress = true
        captureAttemptIndex = 0
        captureAttempts = buildPhotoAttempts(width, height, quality.coerceIn(0, 100))
        requestCurrentPhotoAttempt(captureGeneration)

        Log.d(appTag, "=== Take Photo Request Complete ===")
        Log.d(appTag, "Note: Photo result will be delivered via Hi Rokid image callback")
    }

    private fun requestCurrentPhotoAttempt(generation: Int) {
        if (generation != captureGeneration || !captureInProgress) return

        val request = captureAttempts.getOrNull(captureAttemptIndex)
        if (request == null) {
            failPhotoCapture("Photo capture failed for all supported resolutions")
            return
        }

        Log.d(appTag, "Calling takePhoto through Hi Rokid channel: ${request.width}x${request.height}, Q:${request.quality}")
        RokidHostConnection.setImageStreamListener(imageStreamCallback)
        val requested = RokidHostConnection.takePhoto(request.width, request.height, request.quality)
        Log.d(appTag, "Take photo request success: $requested")

        if (requested) {
            Log.i(appTag, "Photo capture request sent successfully - waiting for callback")
            listener?.onPhotoStatusUpdate("Capturing photo... (${request.width}x${request.height}, Q:${request.quality})")
            handler.postDelayed({
                onPhotoTimeout(generation, request)
            }, PHOTO_CAPTURE_TIMEOUT_MS)
        } else {
            Log.w(appTag, "Photo request rejected for ${request.width}x${request.height}; trying fallback")
            captureAttemptIndex += 1
            requestCurrentPhotoAttempt(generation)
        }
    }

    private fun onPhotoTimeout(generation: Int, request: PhotoRequest) {
        if (generation != captureGeneration || !captureInProgress) return
        Log.w(appTag, "Photo callback timed out for ${request.width}x${request.height}")
        retryOrFail("Photo callback timed out")
    }

    private fun retryOrFail(reason: String) {
        if (!captureInProgress) {
            Log.w(appTag, "Ignoring photo failure after capture completed: $reason")
            return
        }
        captureAttemptIndex += 1
        if (captureAttemptIndex < captureAttempts.size) {
            Log.w(appTag, "$reason; trying next photo fallback")
            requestCurrentPhotoAttempt(captureGeneration)
        } else {
            failPhotoCapture(reason)
        }
    }

    private fun completePhotoCapture() {
        captureInProgress = false
        captureGeneration += 1
        captureAttempts = emptyList()
        captureAttemptIndex = 0
    }

    private fun failPhotoCapture(reason: String) {
        Log.e(appTag, reason)
        completePhotoCapture()
        listener?.onPhotoFailed(reason)
    }

    private fun buildPhotoAttempts(width: Int, height: Int, quality: Int): List<PhotoRequest> {
        val requested = PhotoRequest(width, height, quality)
        val first = if (isSupportedResolution(requested.width, requested.height)) {
            requested
        } else {
            Log.w(appTag, "Unsupported photo size ${requested.width}x${requested.height}; using SDK-safe default")
            PhotoRequest(DEFAULT_PHOTO_WIDTH, DEFAULT_PHOTO_HEIGHT, quality)
        }

        return listOf(
            first,
            PhotoRequest(720, 1280, quality),
            PhotoRequest(640, 480, minOf(quality, 75)),
            PhotoRequest(480, 640, minOf(quality, 75)),
            PhotoRequest(320, 240, minOf(quality, 70))
        ).distinctBy { "${it.width}x${it.height}:${it.quality}" }
    }

    private fun isSupportedResolution(width: Int, height: Int): Boolean =
        SUPPORTED_AI_PHOTO_SIZES.contains(width to height)

    /**
     * Release camera resources
     */
    fun release() {
        captureInProgress = false
        captureGeneration += 1
        captureAttempts = emptyList()
        captureAttemptIndex = 0
        if (isCameraOpen) {
            Log.d(appTag, "Releasing camera resources")
            isCameraOpen = false
        }
        RokidHostConnection.setImageStreamListener(null)
    }

    private data class PhotoRequest(
        val width: Int,
        val height: Int,
        val quality: Int
    )

    private companion object {
        private const val DEFAULT_PHOTO_WIDTH = 1280
        private const val DEFAULT_PHOTO_HEIGHT = 720
        private const val DEFAULT_PHOTO_QUALITY = 75
        private const val PHOTO_CAPTURE_TIMEOUT_MS = 12_000L

        private val SUPPORTED_AI_PHOTO_SIZES = setOf(
            4032 to 3024,
            4000 to 3000,
            4032 to 2268,
            3264 to 2448,
            3200 to 2400,
            2268 to 3024,
            2876 to 2156,
            2688 to 2016,
            2582 to 1936,
            2400 to 1800,
            1800 to 2400,
            2560 to 1440,
            2400 to 1350,
            2048 to 1536,
            2016 to 1512,
            1920 to 1080,
            1600 to 1200,
            1440 to 1080,
            1280 to 720,
            720 to 1280,
            1024 to 768,
            800 to 600,
            648 to 648,
            854 to 480,
            800 to 480,
            640 to 480,
            480 to 640,
            352 to 288,
            320 to 240,
            320 to 180,
            176 to 144
        )
    }
}
