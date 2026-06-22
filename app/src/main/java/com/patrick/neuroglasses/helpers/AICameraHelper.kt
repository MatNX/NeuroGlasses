package com.patrick.neuroglasses.helpers

import android.graphics.Bitmap
import android.graphics.BitmapFactory
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
            listener?.onPhotoFailed(msg ?: "Photo failed: $code")
        }
    }

    private fun handlePhotoData(photo: ByteArray) {
        Log.d(appTag, "=== Photo Result Callback Triggered ===")
        Log.d(appTag, "Photo data size: ${photo.size} bytes")

        if (photo.isEmpty()) {
            Log.w(appTag, "Photo captured but data is empty!")
            listener?.onPhotoFailed("Photo captured but no data received")
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

                listener?.onPhotoSuccess(bitmap, photo.size, bitmap.width, bitmap.height)
            } else {
                Log.e(appTag, "Failed to decode WebP - BitmapFactory returned null")
                listener?.onPhotoFailed("Photo received but decode failed")
            }
        } catch (e: Exception) {
            Log.e(appTag, "Exception while decoding WebP image", e)
            Log.e(appTag, "Exception type: ${e.javaClass.simpleName}")
            Log.e(appTag, "Exception message: ${e.message}")
            listener?.onPhotoFailed("Photo decode error: ${e.message}")
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
    fun openGlassCamera(width: Int = 1920, height: Int = 1080, quality: Int = 85) {
        Log.d(appTag, "=== Opening Glass Camera ===")
        Log.d(appTag, "Camera parameters - Width: $width, Height: $height, Quality: $quality")

        if (RokidHostConnection.isConnected()) {
            Log.i(appTag, "Camera ready through Hi Rokid channel")
            isCameraOpen = true
            RokidHostConnection.setImageStreamListener(imageStreamCallback)
            listener?.onCameraOpened("Camera ready (${width}x${height}, Q:$quality)")
        } else {
            Log.e(appTag, "Hi Rokid channel is not connected")
            isCameraOpen = false
            listener?.onCameraOpenFailed("Hi Rokid channel is not connected")
        }

        Log.d(appTag, "=== Open Camera Complete ===")
    }

    /**
     * Take a photo using the glass camera in AI scene. Note: Due to SDK issue, the width and height is reversed
     * @param width photo width (default 1920)
     * @param height photo height (default 1080)
     * @param quality photo quality 0-100 (default 85)
     */
    fun takePhoto(width: Int = 1080, height: Int = 1920, quality: Int = 85) {
        Log.d(appTag, "=== Take Photo Requested ===")
        Log.d(appTag, "Photo parameters - Width: $width, Height: $height, Quality: $quality")
        Log.d(appTag, "Camera open: $isCameraOpen")

        // Optionally open camera first if not already open
        if (!isCameraOpen) {
            Log.i(appTag, "Camera not open, opening camera first...")
            openGlassCamera(width, height, quality)
        }

        // Take the photo
        if (!isCameraOpen) {
            Log.e(appTag, "Photo capture aborted because camera/channel is not ready")
            listener?.onPhotoFailed("Hi Rokid channel is not connected")
            return
        }

        Log.d(appTag, "Calling takePhoto through Hi Rokid channel...")
        RokidHostConnection.setImageStreamListener(imageStreamCallback)
        val requested = RokidHostConnection.takePhoto(width, height, quality)

        Log.d(appTag, "Take photo request success: $requested")

        if (requested) {
            Log.i(appTag, "Photo capture request sent successfully - waiting for callback")
            listener?.onPhotoStatusUpdate("Capturing photo... (${width}x${height}, Q:$quality)")
        } else {
            Log.e(appTag, "Photo capture request failed!")
            listener?.onPhotoFailed("Photo capture failed")
        }

        Log.d(appTag, "=== Take Photo Request Complete ===")
        Log.d(appTag, "Note: Photo result will be delivered via Hi Rokid image callback")
    }

    /**
     * Release camera resources
     */
    fun release() {
        if (isCameraOpen) {
            Log.d(appTag, "Releasing camera resources")
            isCameraOpen = false
        }
        RokidHostConnection.setImageStreamListener(null)
    }
}
