package com.patrick.neuroglasses.helpers

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import com.example.cxrglobal.callbacks.IAudioStreamCbk
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Helper class for audio recording operations from Rokid glasses.
 * Handles audio stream capture, buffering, and file saving.
 */
class AudioHelper(private val context: Context, private val appTag: String = "AudioHelper") {

    enum class StopReason {
        MANUAL,
        SILENCE_AFTER_SPEECH,
        NO_SPEECH_TIMEOUT,
        REMOTE,
        ERROR
    }

    // Audio recording state
    var isRecording = false
        private set

    var lastStopReason: StopReason = StopReason.MANUAL
        private set

    // Audio recording variables
    private val audioChunks = mutableListOf<ByteArray>()
    private var currentCodecType: Int = 1 // Default to PCM
    private var currentStreamType: String = "AI_assistant"
    private var totalAudioBytes: Long = 0
    private val autoStopHandler = Handler(Looper.getMainLooper())
    private var autoStopOnSilence = false
    private var autoStopping = false
    private var isClosingAudioRecord = false
    private var recordingStartMs = 0L
    private var lastSpeechMs = 0L
    private var hasDetectedSpeech = false
    private var speechFrameCount = 0

    private companion object {
        private const val AUTO_STOP_CHECK_INTERVAL_MS = 250L
        private const val MIN_RECORDING_MS = 900L
        private const val SILENCE_AFTER_SPEECH_MS = 1500L
        private const val NO_SPEECH_TIMEOUT_MS = 15000L
        private const val MIN_SPEECH_FRAMES = 3
        private const val SPEECH_RMS_THRESHOLD = 650.0
        private const val SPEECH_PEAK_THRESHOLD = 2000
    }

    // Track the most recently saved audio file path
    var lastSavedAudioPath: String? = null
        private set

    // Callback interface for audio events
    interface AudioRecordingListener {
        fun onAudioStreamStarted(codecType: Int, streamType: String?)
        fun onAudioDataReceived(chunkSize: Int, totalChunks: Int, totalBytes: Long)
        fun onAudioRecordingStarted(message: String)
        fun onAudioRecordingFailed(message: String)
        fun onAudioRecordingStopped(savedFilePath: String?)
        fun onAudioStatusUpdate(message: String)
    }

    private var listener: AudioRecordingListener? = null

    fun setListener(listener: AudioRecordingListener) {
        this.listener = listener
    }

    /**
     * Audio stream listener to receive audio data from glasses
     */
    private val audioStreamListener = object : IAudioStreamCbk {
        override fun onAudioStreamStateChanged(started: Boolean) {
            Log.d(appTag, "Audio stream state changed: started=$started")
            if (started) {
                markRecordingStarted("Audio recording started")
            } else if (isRecording) {
                if (!isClosingAudioRecord) {
                    Log.i(appTag, "Audio stream stopped remotely")
                    lastStopReason = StopReason.REMOTE
                    finishStoppedRecording()
                }
            }
        }

        override fun onAudioReceived(data: ByteArray, offset: Int, length: Int) {
            Log.d(appTag, "Audio data received - Offset: $offset, Length: $length")

            if (!isRecording) {
                Log.v(appTag, "Ignoring audio chunk because no recording is active")
                return
            }

            if (length <= 0) return

            Log.i(appTag, "Audio chunk: $length bytes")

            // Extract the actual audio data from the chunk
            val audioChunk = ByteArray(length)
            System.arraycopy(data, offset, audioChunk, 0, length)

            // Add to our list of chunks
            audioChunks.add(audioChunk)
            totalAudioBytes += length
            updateVoiceActivity(audioChunk)

            Log.d(appTag, "Chunk stored. Total chunks: ${audioChunks.size}, Total bytes: $totalAudioBytes")

            listener?.onAudioDataReceived(length, audioChunks.size, totalAudioBytes)
        }

        override fun onAudioError(code: Int, msg: String?) {
            Log.e(appTag, "Audio stream error: $code $msg")
            lastStopReason = StopReason.ERROR
            isClosingAudioRecord = false
            if (audioChunks.isNotEmpty()) {
                Log.w(appTag, "Audio stream failed after receiving data; saving buffered audio before reconnect")
                finishStoppedRecording()
            } else {
                cancelAutoStopCheck()
                isRecording = false
                listener?.onAudioRecordingFailed(msg ?: "Audio stream error $code")
            }
        }
    }

    /**
     * Set or remove the audio stream listener
     * @param set true: set the listener, false: remove the listener
     */
    fun setAudioStreamListener(set: Boolean) {
        RokidHostConnection.setAudioStreamListener(if (set) audioStreamListener else null)
        Log.d(appTag, "Audio stream listener ${if (set) "set" else "removed"}")
    }

    /**
     * Open audio recording from glasses
     * @param codecType The stream codec type: 1:pcm, 2:opus
     * @param streamType The stream type identifier (e.g., "AI_assistant")
     * @param autoStopOnSilence Automatically stop PCM recording after the user stops speaking.
     * @return The status of the open audio record request
     */
    fun openAudioRecord(
        codecType: Int = 1,
        streamType: String = "AI_assistant",
        autoStopOnSilence: Boolean = true
    ): Boolean {
        Log.d(appTag, "=== Opening Audio Record ===")
        Log.d(appTag, "Codec type: ${if (codecType == 1) "PCM (1)" else if (codecType == 2) "OPUS (2)" else "Unknown ($codecType)"}")
        Log.d(appTag, "Stream type: $streamType")

        currentCodecType = codecType
        currentStreamType = streamType
        audioChunks.clear()
        totalAudioBytes = 0
        this.autoStopOnSilence = autoStopOnSilence && codecType == 1
        autoStopping = false
        isClosingAudioRecord = false
        hasDetectedSpeech = false
        speechFrameCount = 0
        lastStopReason = StopReason.MANUAL
        recordingStartMs = SystemClock.elapsedRealtime()
        lastSpeechMs = recordingStartMs

        val started = RokidHostConnection.startAudioStream(codecType)
        Log.d(appTag, "Open audio record success: $started")

        if (started) {
            markRecordingStarted("Audio recording started")
            scheduleAutoStopCheck()
        } else {
            Log.e(appTag, "Failed to start audio recording")
            listener?.onAudioRecordingFailed("Audio recording failed")
        }

        Log.d(appTag, "=== Open Audio Record Complete ===")
        return started
    }

    private fun markRecordingStarted(message: String) {
        if (isRecording) return
        Log.i(appTag, "Audio recording started successfully")
        Log.d(appTag, "Codec type: ${if (currentCodecType == 1) "PCM" else if (currentCodecType == 2) "OPUS" else "Unknown ($currentCodecType)"}")
        Log.d(appTag, "Stream type: $currentStreamType")
        isRecording = true
        listener?.onAudioStreamStarted(currentCodecType, currentStreamType)
        listener?.onAudioRecordingStarted(message)
    }

    private val autoStopRunnable = object : Runnable {
        override fun run() {
            if (!isRecording || !autoStopOnSilence || autoStopping) return

            val now = SystemClock.elapsedRealtime()
            val elapsed = now - recordingStartMs
            val quietFor = now - lastSpeechMs
            val shouldStop = if (hasDetectedSpeech) {
                elapsed >= MIN_RECORDING_MS && quietFor >= SILENCE_AFTER_SPEECH_MS
            } else {
                elapsed >= NO_SPEECH_TIMEOUT_MS
            }

            if (shouldStop) {
                autoStopping = true
                val reason = if (hasDetectedSpeech) {
                    lastStopReason = StopReason.SILENCE_AFTER_SPEECH
                    "Silence detected after speech"
                } else {
                    lastStopReason = StopReason.NO_SPEECH_TIMEOUT
                    "No speech detected"
                }
                Log.i(appTag, "$reason; stopping audio recording")
                listener?.onAudioStatusUpdate(reason)
                closeAudioRecord(currentStreamType)
                return
            }

            autoStopHandler.postDelayed(this, AUTO_STOP_CHECK_INTERVAL_MS)
        }
    }

    private fun scheduleAutoStopCheck() {
        autoStopHandler.removeCallbacks(autoStopRunnable)
        if (autoStopOnSilence) {
            autoStopHandler.postDelayed(autoStopRunnable, AUTO_STOP_CHECK_INTERVAL_MS)
        }
    }

    private fun cancelAutoStopCheck() {
        autoStopHandler.removeCallbacks(autoStopRunnable)
        autoStopping = false
    }

    private fun updateVoiceActivity(audioChunk: ByteArray) {
        if (!autoStopOnSilence || currentCodecType != 1) return
        if (isLikelySpeech(audioChunk)) {
            speechFrameCount++
            if (speechFrameCount >= MIN_SPEECH_FRAMES) {
                hasDetectedSpeech = true
                lastSpeechMs = SystemClock.elapsedRealtime()
            }
        }
    }

    private fun isLikelySpeech(audioChunk: ByteArray): Boolean {
        if (audioChunk.size < 2) return false

        var index = 0
        var samples = 0
        var peak = 0
        var sumSquares = 0.0

        while (index + 1 < audioChunk.size) {
            val sample = ((audioChunk[index].toInt() and 0xFF) or (audioChunk[index + 1].toInt() shl 8)).toShort().toInt()
            val magnitude = abs(sample)
            if (magnitude > peak) peak = magnitude
            sumSquares += sample.toDouble() * sample.toDouble()
            samples++
            index += 2
        }

        if (samples == 0) return false
        val rms = sqrt(sumSquares / samples)
        return rms >= SPEECH_RMS_THRESHOLD || peak >= SPEECH_PEAK_THRESHOLD
    }

    /**
     * Write WAV header to the output stream
     * WAV format specification for PCM audio
     */
    private fun writeWavHeader(outputStream: FileOutputStream, dataSize: Long) {
        val sampleRate = 16000 // 16kHz sample rate (typical for Rokid glasses)
        val channels = 1 // Mono audio
        val bitsPerSample = 16 // 16-bit PCM

        val byteRate = sampleRate * channels * bitsPerSample / 8
        val blockAlign = channels * bitsPerSample / 8

        // RIFF header
        outputStream.write("RIFF".toByteArray())
        outputStream.write(intToBytes((dataSize + 36).toInt())) // File size - 8
        outputStream.write("WAVE".toByteArray())

        // fmt subchunk
        outputStream.write("fmt ".toByteArray())
        outputStream.write(intToBytes(16)) // Subchunk1Size (16 for PCM)
        outputStream.write(shortToBytes(1)) // AudioFormat (1 for PCM)
        outputStream.write(shortToBytes(channels.toShort())) // NumChannels
        outputStream.write(intToBytes(sampleRate)) // SampleRate
        outputStream.write(intToBytes(byteRate)) // ByteRate
        outputStream.write(shortToBytes(blockAlign.toShort())) // BlockAlign
        outputStream.write(shortToBytes(bitsPerSample.toShort())) // BitsPerSample

        // data subchunk
        outputStream.write("data".toByteArray())
        outputStream.write(intToBytes(dataSize.toInt())) // Subchunk2Size
    }

    /**
     * Convert integer to little-endian byte array
     */
    private fun intToBytes(value: Int): ByteArray {
        return byteArrayOf(
            (value and 0xFF).toByte(),
            ((value shr 8) and 0xFF).toByte(),
            ((value shr 16) and 0xFF).toByte(),
            ((value shr 24) and 0xFF).toByte()
        )
    }

    /**
     * Convert short to little-endian byte array
     */
    private fun shortToBytes(value: Short): ByteArray {
        return byteArrayOf(
            (value.toInt() and 0xFF).toByte(),
            ((value.toInt() shr 8) and 0xFF).toByte()
        )
    }

    /**
     * Save accumulated audio chunks to a file
     * @return The file path if successful, null otherwise
     */
    fun saveAudioToFile(): String? {
        if (audioChunks.isEmpty()) {
            Log.w(appTag, "No audio data to save")
            return null
        }

        try {
            Log.d(appTag, "=== Saving Audio to File ===")
            Log.d(appTag, "Total chunks to save: ${audioChunks.size}")
            Log.d(appTag, "Total bytes to save: $totalAudioBytes")

            // Create timestamp for unique filename
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())

            // Determine file extension based on codec
            val fileExtension = when (currentCodecType) {
                1 -> "wav"  // PCM saved as WAV for easy playback
                2 -> "opus" // OPUS audio
                else -> "raw"
            }

            val fileName = "audio_${timestamp}.$fileExtension"

            // Save to app's external files directory (no permissions needed)
            val audioDir = File(context.getExternalFilesDir(null), "audio_recordings")
            if (!audioDir.exists()) {
                audioDir.mkdirs()
                Log.d(appTag, "Created audio directory: ${audioDir.absolutePath}")
            }

            val audioFile = File(audioDir, fileName)
            Log.i(appTag, "Saving audio to: ${audioFile.absolutePath}")

            // Write all chunks to file
            FileOutputStream(audioFile).use { outputStream ->
                // Write WAV header for PCM audio
                if (currentCodecType == 1) {
                    writeWavHeader(outputStream, totalAudioBytes)
                }

                var bytesWritten = 0L
                audioChunks.forEach { chunk ->
                    outputStream.write(chunk)
                    bytesWritten += chunk.size
                }
                Log.d(appTag, "Bytes written: $bytesWritten")
            }

            val fileSizeKB = audioFile.length() / 1024.0
            Log.i(appTag, "Audio file saved successfully!")
            Log.i(appTag, "File size: ${String.format("%.2f", fileSizeKB)} KB")
            Log.i(appTag, "File path: ${audioFile.absolutePath}")

            // Clear chunks after saving
            audioChunks.clear()
            totalAudioBytes = 0
            Log.d(appTag, "Audio buffer cleared")

            // Store the path of the most recently saved file
            lastSavedAudioPath = audioFile.absolutePath

            return audioFile.absolutePath

        } catch (e: Exception) {
            Log.e(appTag, "Error saving audio file", e)
            Log.e(appTag, "Exception type: ${e.javaClass.simpleName}")
            Log.e(appTag, "Exception message: ${e.message}")
            return null
        }
    }

    /**
     * Close audio recording from glasses
     * @param streamType The stream type identifier (must match the one used in openAudioRecord)
     * @return The status of the close audio record request
     */
    fun closeAudioRecord(streamType: String = "AI_assistant"): Boolean {
        Log.d(appTag, "=== Closing Audio Record ===")
        Log.d(appTag, "Stream type: $streamType")
        if (!autoStopping) {
            lastStopReason = StopReason.MANUAL
        }
        cancelAutoStopCheck()

        isClosingAudioRecord = true
        val stopped = RokidHostConnection.stopAudioStream()
        isClosingAudioRecord = false
        Log.d(appTag, "Close audio record success: $stopped")

        if (stopped) {
            Log.i(appTag, "Audio recording stopped successfully")
            finishStoppedRecording()
        } else if (audioChunks.isNotEmpty()) {
            Log.w(appTag, "Audio stop failed after receiving data; saving buffered audio anyway")
            finishStoppedRecording()
        } else {
            Log.e(appTag, "Failed to stop audio recording")
            isRecording = false
            listener?.onAudioRecordingFailed("Audio stop failed")
        }

        Log.d(appTag, "=== Close Audio Record Complete ===")
        return stopped
    }

    /**
     * Abort a bad/stale recording without saving or notifying a completed recording.
     * Used when the SDK stream gets stuck between stop and start during reconnect.
     */
    fun abortRecordingForReconnect(): Boolean {
        Log.d(appTag, "=== Aborting Audio Record For Reconnect ===")
        cancelAutoStopCheck()
        isClosingAudioRecord = true
        val stopped = runCatching { RokidHostConnection.stopAudioStream() }.getOrDefault(false)
        isClosingAudioRecord = false
        isRecording = false
        lastStopReason = StopReason.ERROR
        clearAudioCache()
        Log.d(appTag, "Abort audio record stop sent: $stopped")
        Log.d(appTag, "=== Abort Audio Record Complete ===")
        return stopped
    }

    /**
     * Force-stop the Rokid audio stream during a user/session close.
     * This is intentionally unconditional because the SDK can keep the
     * microphone channel open even after our local recording flag fell behind.
     */
    fun abortRecordingForSessionClose(): Boolean {
        Log.d(appTag, "=== Aborting Audio Record For Session Close ===")
        cancelAutoStopCheck()
        isClosingAudioRecord = true
        val stopped = runCatching { RokidHostConnection.stopAudioStream() }.getOrDefault(false)
        isClosingAudioRecord = false
        isRecording = false
        lastStopReason = StopReason.MANUAL
        clearAudioCache()
        Log.d(appTag, "Session-close audio stop sent: $stopped")
        Log.d(appTag, "=== Abort Audio Record For Session Close Complete ===")
        return stopped
    }

    private fun finishStoppedRecording() {
        cancelAutoStopCheck()
        isRecording = false

        val savedFilePath = if (lastStopReason == StopReason.NO_SPEECH_TIMEOUT) {
            Log.i(appTag, "No-speech timeout reached; discarding buffered audio")
            clearAudioCache()
            null
        } else {
            // Save the recorded audio to file
            saveAudioToFile()
        }

        if (savedFilePath != null) {
            Log.i(appTag, "Audio file notification: $savedFilePath")
        } else {
            if (audioChunks.isEmpty()) {
                Log.w(appTag, "No audio data recorded")
            } else {
                Log.e(appTag, "Failed to save audio file")
            }
        }

        listener?.onAudioRecordingStopped(savedFilePath)
    }

    /**
     * Clear audio cache without saving
     * Used when you want to discard recorded audio
     */
    fun clearAudioCache() {
        Log.d(appTag, "=== Clearing Audio Cache ===")
        Log.d(appTag, "Discarding ${audioChunks.size} chunks, $totalAudioBytes bytes")
        audioChunks.clear()
        totalAudioBytes = 0
        Log.d(appTag, "Audio cache cleared")
    }

    /**
     * Get current recording statistics
     */
    fun getRecordingStats(): RecordingStats {
        return RecordingStats(
            isRecording = isRecording,
            totalChunks = audioChunks.size,
            totalBytes = totalAudioBytes,
            codecType = currentCodecType
        )
    }

    /**
     * Data class for recording statistics
     */
    data class RecordingStats(
        val isRecording: Boolean,
        val totalChunks: Int,
        val totalBytes: Long,
        val codecType: Int
    )

    /**
     * Release audio resources and remove listeners
     */
    fun release() {
        if (isRecording) {
            Log.d(appTag, "Releasing audio resources - stopping active recording")
            closeAudioRecord()
        }
        setAudioStreamListener(false)
        cancelAutoStopCheck()
        audioChunks.clear()
        totalAudioBytes = 0
    }
}
