package com.patrick.neuroglasses.helpers

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaPlayer
import android.os.SystemClock
import android.util.Base64
import android.util.Log
import com.example.cxrglobal.callbacks.ICustomViewCbk
import com.patrick.neuroglasses.R
import java.io.ByteArrayOutputStream
import java.io.File
import androidx.core.graphics.scale
import org.json.JSONArray
import org.json.JSONObject

/**
 * Custom Scene Helper
 * Handles custom UI display on AR glasses
 */
class CustomSceneHelper(
    private val context: Context,
    private val appTag: String = "CustomSceneHelper"
) {

    /**
     * Listener interface for custom view events
     */
    interface CustomSceneListener {
        /**
         * Called when custom view is successfully opened
         */
        fun onSceneOpened()

        /**
         * Called when custom view fails to open
         * @param errorCode The error code
         */
        fun onSceneOpenFailed(errorCode: Int)

        /**
         * Called when custom view is closed
         */
        fun onSceneClosed()

        /**
         * Called when custom view is updated
         */
        fun onSceneUpdated()
    }

    private var listener: CustomSceneListener? = null
    private var isCustomViewListenerSet = false
    private var mediaPlayer: MediaPlayer? = null
    private var audioFileToPlay: File? = null
    private var lastVisibleSceneEventMs = 0L

    // Icon variables
    private var iconsSent = false
    private val aiIconName = "ai_icon"

    /**
     * Set the listener for custom scene events
     */
    fun setListener(listener: CustomSceneListener) {
        this.listener = listener
    }

    /**
     * Initialize the custom view listener
     * Should be called once during setup
     */
    fun initializeCustomViewListener() {
        if (!isCustomViewListenerSet) {
            RokidHostConnection.setCustomViewListener(object : ICustomViewCbk {
                override fun onCustomViewIconsSent() {
                    Log.d(appTag, "Custom view icons sent")
                }

                override fun onCustomViewOpened() {
                    Log.d(appTag, "Custom view opened")
                    lastVisibleSceneEventMs = SystemClock.elapsedRealtime()
                    listener?.onSceneOpened()

                    // Play audio if available
                    audioFileToPlay?.let { playAudio(it) }
                }

                override fun onCustomViewError(code: Int, msg: String?) {
                    if (SystemClock.elapsedRealtime() - lastVisibleSceneEventMs < CUSTOM_VIEW_FALSE_ERROR_GRACE_MS) {
                        Log.w(appTag, "Ignoring custom view error after visible scene/update: $code $msg")
                        return
                    }
                    Log.e(appTag, "Custom view error: $code $msg")
                    listener?.onSceneOpenFailed(code)
                }

                override fun onCustomViewUpdated() {
                    Log.d(appTag, "Custom view updated")
                    lastVisibleSceneEventMs = SystemClock.elapsedRealtime()
                    listener?.onSceneUpdated()
                }

                override fun onCustomViewClosed() {
                    Log.d(appTag, "Custom view closed")
                    listener?.onSceneClosed()
                }
            })
            isCustomViewListenerSet = true
            Log.d(appTag, "Custom view listener initialized")
        }
    }

    /**
     * Set the audio file to play when custom view opens
     * @param audioFile The audio file to play
     */
    fun setAudioFile(audioFile: File) {
        audioFileToPlay = audioFile
        Log.d(appTag, "Audio file set: ${audioFile.absolutePath}")
    }

    /**
     * Play audio file
     * @param audioFile The audio file to play
     */
    fun playAudio(audioFile: File) {
        if (!audioFile.exists()) {
            Log.e(appTag, "Audio file does not exist: ${audioFile.path}")
            return
        }

        try {
            // Stop any currently playing audio
            stopAudio()

            // Create and configure MediaPlayer
            mediaPlayer = MediaPlayer().apply {
                setDataSource(audioFile.absolutePath)
                prepare()

                // Set completion listener
                setOnCompletionListener {
                    Log.d(appTag, "Audio playback completed")
                    stopAudio()
                }

                // Set error listener
                setOnErrorListener { _, what, extra ->
                    Log.e(appTag, "MediaPlayer error: what=$what, extra=$extra")
                    stopAudio()
                    true
                }

                // Start playback
                start()
            }

            Log.i(appTag, "Started playing audio: ${audioFile.name}")
        } catch (e: Exception) {
            Log.e(appTag, "Error playing audio file: ${e.message}", e)
            stopAudio()
        }
    }

    /**
     * Stop audio playback
     */
    fun stopAudio() {
        mediaPlayer?.apply {
            try {
                if (isPlaying) {
                    stop()
                    Log.d(appTag, "Audio playback stopped")
                }
            } catch (e: IllegalStateException) {
                Log.w(appTag, "MediaPlayer already stopped: ${e.message}")
            }
            release()
        }
        mediaPlayer = null
        audioFileToPlay = null
    }

    /**
     * Send AI icon to glasses
     * @return The status of the request
     */
    private fun sendAiIcon(): Boolean {
        if (iconsSent) {
            Log.d(appTag, "Icons already sent, skipping")
            return true
        }

        try {
            // Load the AI icon from drawable resources using R identifier
            val bitmap = BitmapFactory.decodeResource(context.resources, R.drawable.ai_icon_small)
            if (bitmap == null) {
                Log.e(appTag, "Failed to decode AI icon bitmap")
                return false
            }

            // Resize bitmap to fit within 128x128 requirement if needed
            val maxSize = 128
            val resizedBitmap = if (bitmap.width > maxSize || bitmap.height > maxSize) {
                val scale = minOf(maxSize.toFloat() / bitmap.width, maxSize.toFloat() / bitmap.height)
                val newWidth = (bitmap.width * scale).toInt()
                val newHeight = (bitmap.height * scale).toInt()
                bitmap.scale(newWidth, newHeight).also {
                    if (it != bitmap) bitmap.recycle()
                }
            } else {
                bitmap
            }

            // Convert bitmap to Base64
            val base64Icon = bitmapToBase64(resizedBitmap)
            resizedBitmap.recycle()

            if (base64Icon == null) {
                Log.e(appTag, "Failed to convert bitmap to Base64")
                return false
            }

            val iconsJson = JSONArray()
                .put(
                    JSONObject()
                        .put("name", aiIconName)
                        .put("data", base64Icon)
                )
                .toString()
            val success = RokidHostConnection.customViewSetIcons(iconsJson)

            if (success) {
                iconsSent = true
                Log.i(appTag, "AI icon sent successfully")
            } else {
                Log.e(appTag, "Failed to send AI icon")
            }

            return success
        } catch (e: Exception) {
            Log.e(appTag, "Error sending AI icon: ${e.message}", e)
            return false
        }
    }

    /**
     * Convert bitmap to Base64 string
     * @param bitmap The bitmap to convert
     * @return Base64 encoded string or null if failed
     */
    private fun bitmapToBase64(bitmap: Bitmap): String? {
        return try {
            val outputStream = ByteArrayOutputStream()
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
            val byteArray = outputStream.toByteArray()
            Base64.encodeToString(byteArray, Base64.NO_WRAP)
        } catch (e: Exception) {
            Log.e(appTag, "Error converting bitmap to Base64: ${e.message}", e)
            null
        }
    }

    /**
     * Display text result on glasses using custom UI
     * @param resultText The text to display
     * @return The status of the request
     */
    fun displayTextResult(resultText: String): Boolean {
        Log.i(appTag, "Displaying text result: $resultText")

        // Keep the text view clear of the icon area; the Rokid renderer can
        // overlap RelativeLayout children during rapid streaming updates.
        val customViewData = """
            {
                "type": "RelativeLayout",
                "props": {
                    "layout_width": "match_parent",
                    "layout_height": "match_parent",
                    "backgroundColor": "#FF000000",
                    "paddingStart": "24dp",
                    "paddingTop": "120dp",
                    "paddingEnd": "24dp",
                    "paddingBottom": "24dp"
                },
                "children": [
                    {
                        "type": "TextView",
                        "props": {
                            "id": "tv_result",
                            "layout_width": "match_parent",
                            "layout_height": "wrap_content",
                            "text": ${JSONObject.quote(resultText)},
                            "textSize": "17sp",
                            "textColor": "#FF00FF00",
                            "textStyle": "bold",
                            "layout_alignParentStart": "true",
                            "layout_alignParentTop": "true"
                        }
                    }
                ]
            }
        """.trimIndent()

        // Open custom UI with result
        val success = RokidHostConnection.customViewOpen(customViewData)
        Log.d(appTag, "Open custom view success: $success")
        if (!success) {
            Log.w(appTag, "customViewOpen returned false; waiting for callback/update before reporting failure")
        }

        return success
    }

    /**
     * Update the displayed text
     * @param newText The new text to display
     * @return The status of the request
     */
    fun updateTextResult(newText: String): Boolean {
        Log.i(appTag, "Updating text result: $newText")

        // Create update JSON
        val updateData = """
            [
                {
                    "action": "update",
                    "id": "tv_result",
                    "props": {
                        "text": ${JSONObject.quote(newText)}
                    }
                }
            ]
        """.trimIndent()

        val success = RokidHostConnection.customViewUpdate(updateData)
        Log.d(appTag, "Update custom view success: $success")
        if (success) {
            lastVisibleSceneEventMs = SystemClock.elapsedRealtime()
        }

        return success
    }

    /**
     * Close the custom view
     * @return The status of the request
     */
    fun closeCustomView(): Boolean {
        Log.d(appTag, "Closing custom view")
        lastVisibleSceneEventMs = 0L
        val success = RokidHostConnection.customViewClose()
        Log.d(appTag, "Close custom view success: $success")
        return success
    }

    /**
     * Release resources and remove listener
     */
    fun release() {
        stopAudio()
        RokidHostConnection.setCustomViewListener(null)
        listener = null
        isCustomViewListenerSet = false
        audioFileToPlay = null
        Log.d(appTag, "CustomSceneHelper released")
    }

    private companion object {
        private const val CUSTOM_VIEW_FALSE_ERROR_GRACE_MS = 1200L
    }
}
