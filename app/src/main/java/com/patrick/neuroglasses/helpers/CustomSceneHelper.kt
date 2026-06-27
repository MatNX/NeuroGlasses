package com.patrick.neuroglasses.helpers

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaPlayer
import android.os.Handler
import android.os.Looper
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
    private var suppressSceneClosedUntilMs = 0L

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
                    iconsSent = true
                }

                override fun onCustomViewOpened() {
                    Log.d(appTag, "Custom view opened")
                    lastVisibleSceneEventMs = SystemClock.elapsedRealtime()
                    if (isTransientPopupVisible()) {
                        Log.d(appTag, "Ignoring custom view opened callback for transient popup")
                        return
                    }
                    listener?.onSceneOpened()

                    // Play audio if available
                    audioFileToPlay?.let { playAudio(it) }
                }

                override fun onCustomViewError(code: Int, msg: String?) {
                    if (isTransientPopupVisible()) {
                        Log.w(appTag, "Ignoring transient popup custom view error: $code $msg")
                        return
                    }
                    if (SystemClock.elapsedRealtime() - lastVisibleSceneEventMs < CUSTOM_VIEW_FALSE_ERROR_GRACE_MS) {
                        Log.w(appTag, "Ignoring custom view error after visible scene/update: $code $msg")
                        return
                    }
                    Log.e(appTag, "Custom view error: $code $msg")
                    iconsSent = false
                    listener?.onSceneOpenFailed(code)
                }

                override fun onCustomViewUpdated() {
                    Log.d(appTag, "Custom view updated")
                    lastVisibleSceneEventMs = SystemClock.elapsedRealtime()
                    if (isTransientPopupVisible()) {
                        Log.d(appTag, "Ignoring custom view updated callback for transient popup")
                        return
                    }
                    listener?.onSceneUpdated()
                }

                override fun onCustomViewClosed() {
                    Log.d(appTag, "Custom view closed")
                    val hadVisibleScene = lastVisibleSceneEventMs > 0L
                    lastVisibleSceneEventMs = 0L
                    if (!hadVisibleScene && shouldSuppressSceneClosedCallback()) {
                        Log.d(appTag, "Ignoring app-initiated custom view close")
                        return
                    }
                    suppressSceneClosedUntilMs = 0L
                    iconsSent = false
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
    private fun sendAiIcon(force: Boolean = false): Boolean {
        if (iconsSent && !force) {
            Log.d(appTag, "Icons already sent, skipping")
            return true
        }

        try {
            // Use the Neuro-sama app artwork for the glasses-side assistant icon.
            val bitmap = BitmapFactory.decodeResource(context.resources, R.drawable.neuro_glasses_icon)
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
        if (deferPersistentViewWhilePopupIsVisible(resultText)) {
            Log.d(appTag, "Persistent text display deferred while transient popup is visible")
            return true
        }
        rememberPersistentView(resultText)

        if (!sendAiIcon(force = true)) {
            Log.w(appTag, "Continuing custom view open even though the AI icon was not confirmed")
        }

        // Keep the text view clear of the icon area; the Rokid renderer can
        // overlap RelativeLayout children during rapid streaming updates.
        val customViewData = """
            {
                "type": "RelativeLayout",
                "props": {
                    "layout_width": "match_parent",
                    "layout_height": "match_parent",
                    "backgroundColor": "#FF000000",
                    "paddingStart": "20dp",
                    "paddingTop": "24dp",
                    "paddingEnd": "24dp",
                    "paddingBottom": "24dp"
                },
                "children": [
                    {
                        "type": "ImageView",
                        "props": {
                            "id": "iv_neuro_icon",
                            "layout_width": "64dp",
                            "layout_height": "64dp",
                            "name": "$aiIconName",
                            "scaleType": "center_inside",
                            "layout_alignParentTop": "true",
                            "layout_alignParentStart": "true"
                        }
                    },
                    {
                        "type": "TextView",
                        "props": {
                            "id": "tv_result",
                            "layout_width": "match_parent",
                            "layout_height": "wrap_content",
                            "text": ${JSONObject.quote(resultText)},
                            "textSize": "14sp",
                            "textColor": "#FF00FF00",
                            "textStyle": "bold",
                            "layout_toEndOf": "iv_neuro_icon",
                            "layout_alignParentEnd": "true",
                            "layout_alignTop": "iv_neuro_icon",
                            "marginStart": "16dp"
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
     * Display a short-lived navigation popup without replacing the assistant view permanently.
     */
    fun displayNavigationPopup(
        message: String,
        durationMs: Long = NAVIGATION_POPUP_DURATION_MS
    ): Boolean {
        val cleanMessage = message.trim()
        if (cleanMessage.isBlank()) return false

        val generation = markTransientPopupVisible(durationMs)
        val customViewData = """
            {
                "type": "RelativeLayout",
                "props": {
                    "layout_width": "match_parent",
                    "layout_height": "match_parent",
                    "backgroundColor": "#00000000",
                    "paddingStart": "18dp",
                    "paddingTop": "18dp",
                    "paddingEnd": "18dp",
                    "paddingBottom": "22dp"
                },
                "children": [
                    {
                        "type": "TextView",
                        "props": {
                            "id": "tv_nav_popup",
                            "layout_width": "match_parent",
                            "layout_height": "wrap_content",
                            "text": ${JSONObject.quote("Navigation\n$cleanMessage")},
                            "textSize": "16sp",
                            "textColor": "#FF00FF00",
                            "textStyle": "bold",
                            "backgroundColor": "#DD000000",
                            "paddingStart": "18dp",
                            "paddingTop": "14dp",
                            "paddingEnd": "18dp",
                            "paddingBottom": "14dp",
                            "layout_alignParentBottom": "true",
                            "layout_alignParentStart": "true",
                            "layout_alignParentEnd": "true"
                        }
                    }
                ]
            }
        """.trimIndent()

        val success = RokidHostConnection.customViewOpen(customViewData)
        Log.d(appTag, "Open navigation popup success: $success")
        if (!success) {
            clearTransientPopup(generation)
            return false
        }

        sharedHandler.postDelayed({
            dismissTransientPopup(generation)
        }, durationMs)
        return true
    }

    /**
     * Update the displayed text
     * @param newText The new text to display
     * @return The status of the request
     */
    fun updateTextResult(newText: String): Boolean {
        Log.i(appTag, "Updating text result: $newText")
        if (deferPersistentViewWhilePopupIsVisible(newText)) {
            Log.d(appTag, "Persistent text update deferred while transient popup is visible")
            return true
        }
        rememberPersistentView(newText)

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
    fun closeCustomView(suppressSceneClosedCallback: Boolean = true): Boolean {
        Log.d(appTag, "Closing custom view")
        clearPersistentView()
        clearTransientPopup()
        lastVisibleSceneEventMs = 0L
        val requestedAtMs = SystemClock.elapsedRealtime()
        if (suppressSceneClosedCallback) {
            suppressSceneClosedUntilMs = requestedAtMs + APP_CLOSE_EVENT_SUPPRESSION_MS
            suppressSceneClosedGlobally(APP_CLOSE_EVENT_SUPPRESSION_MS)
        }
        val success = RokidHostConnection.customViewClose()
        if (!success && suppressSceneClosedCallback) {
            suppressSceneClosedUntilMs = requestedAtMs + APP_CLOSE_EVENT_FALSE_SUPPRESSION_MS
            suppressSceneClosedGlobally(APP_CLOSE_EVENT_FALSE_SUPPRESSION_MS)
        }
        Log.d(appTag, "Close custom view success: $success")
        return success
    }

    fun isCustomViewOpen(): Boolean =
        RokidHostConnection.customViewIsOpen()

    private fun shouldSuppressSceneClosedCallback(): Boolean {
        val now = SystemClock.elapsedRealtime()
        val shouldSuppress = (suppressSceneClosedUntilMs > 0L && now <= suppressSceneClosedUntilMs) ||
            consumeGlobalSceneClosedSuppression(now)
        suppressSceneClosedUntilMs = 0L
        return shouldSuppress
    }

    private fun deferPersistentViewWhilePopupIsVisible(text: String): Boolean {
        synchronized(sceneStateLock) {
            val now = SystemClock.elapsedRealtime()
            if (now > transientPopupUntilMs) return false
            persistentViewVisible = true
            persistentViewText = text
            return true
        }
    }

    private fun rememberPersistentView(text: String) {
        synchronized(sceneStateLock) {
            persistentViewVisible = true
            persistentViewText = text
        }
    }

    private fun clearPersistentView() {
        synchronized(sceneStateLock) {
            persistentViewVisible = false
            persistentViewText = null
        }
    }

    private fun markTransientPopupVisible(durationMs: Long): Int {
        synchronized(sceneStateLock) {
            transientPopupGeneration += 1
            transientPopupUntilMs = SystemClock.elapsedRealtime() + durationMs
            return transientPopupGeneration
        }
    }

    private fun clearTransientPopup(generation: Int? = null) {
        synchronized(sceneStateLock) {
            if (generation == null || generation == transientPopupGeneration) {
                transientPopupUntilMs = 0L
            }
        }
    }

    private fun dismissTransientPopup(generation: Int) {
        val shouldRestore = synchronized(sceneStateLock) {
            if (generation != transientPopupGeneration) return
            transientPopupUntilMs = 0L
            persistentViewVisible && !persistentViewText.isNullOrBlank()
        }

        suppressSceneClosedGlobally(APP_CLOSE_EVENT_SUPPRESSION_MS)
        val success = RokidHostConnection.customViewClose()
        Log.d(appTag, "Dismiss navigation popup success: $success")

        if (shouldRestore) {
            sharedHandler.postDelayed({
                restorePersistentViewAfterPopup(generation)
            }, NAVIGATION_POPUP_RESTORE_DELAY_MS)
        }
    }

    private fun restorePersistentViewAfterPopup(generation: Int) {
        val textToRestore = synchronized(sceneStateLock) {
            if (generation != transientPopupGeneration) return
            if (!persistentViewVisible) return
            persistentViewText
        } ?: return

        Log.d(appTag, "Restoring persistent custom view after navigation popup")
        displayTextResult(textToRestore)
    }

    /**
     * Release resources and remove listener
     */
    fun release() {
        stopAudio()
        RokidHostConnection.setCustomViewListener(null)
        listener = null
        isCustomViewListenerSet = false
        iconsSent = false
        audioFileToPlay = null
        Log.d(appTag, "CustomSceneHelper released")
    }

    private companion object {
        private const val CUSTOM_VIEW_FALSE_ERROR_GRACE_MS = 1200L
        private const val APP_CLOSE_EVENT_SUPPRESSION_MS = 1000L
        private const val APP_CLOSE_EVENT_FALSE_SUPPRESSION_MS = 250L
        private const val NAVIGATION_POPUP_DURATION_MS = 5_500L
        private const val NAVIGATION_POPUP_RESTORE_DELAY_MS = 180L

        private val sharedHandler = Handler(Looper.getMainLooper())
        private val sceneStateLock = Any()
        private var persistentViewText: String? = null
        private var persistentViewVisible = false
        private var transientPopupUntilMs = 0L
        private var transientPopupGeneration = 0
        private var globalSuppressSceneClosedUntilMs = 0L

        private fun suppressSceneClosedGlobally(durationMs: Long) {
            synchronized(sceneStateLock) {
                globalSuppressSceneClosedUntilMs = maxOf(
                    globalSuppressSceneClosedUntilMs,
                    SystemClock.elapsedRealtime() + durationMs
                )
            }
        }

        private fun consumeGlobalSceneClosedSuppression(now: Long): Boolean {
            synchronized(sceneStateLock) {
                val shouldSuppress = globalSuppressSceneClosedUntilMs > 0L &&
                    now <= globalSuppressSceneClosedUntilMs
                if (shouldSuppress) {
                    globalSuppressSceneClosedUntilMs = 0L
                }
                return shouldSuppress
            }
        }

        private fun isTransientPopupVisible(): Boolean {
            synchronized(sceneStateLock) {
                return SystemClock.elapsedRealtime() <= transientPopupUntilMs
            }
        }
    }
}
