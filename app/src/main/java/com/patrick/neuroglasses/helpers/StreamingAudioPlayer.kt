package com.patrick.neuroglasses.helpers

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.MediaCodec
import android.media.MediaPlayer
import android.media.MediaFormat
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * Streaming audio player for TTS chunks.
 *
 * Groq Orpheus currently returns WAV only, so WAV chunks are streamed directly
 * to AudioTrack after the RIFF header is parsed. The MP3 decoder path remains
 * as a fallback for older or user-configured endpoints.
 */
class StreamingAudioPlayer(private val appTag: String = "StreamingAudioPlayer") {

    private enum class AudioContainer {
        WAV,
        MP3
    }

    private data class WavInfo(
        val sampleRate: Int,
        val channelCount: Int,
        val bitsPerSample: Int,
        val dataStart: Int
    )

    private companion object {
        private const val MIN_PLAYABLE_AUDIO_BYTES = 44L
    }

    private val audioChunkQueue = LinkedBlockingQueue<ByteArray>()

    private var audioTrack: AudioTrack? = null
    private var mediaCodec: MediaCodec? = null
    private var mediaPlayer: MediaPlayer? = null

    @Volatile private var isPlaying = false
    @Volatile private var shouldContinuePlaying = true
    @Volatile private var isStreamComplete = false

    private var playbackThread: Thread? = null
    private var sampleRate = 24000
    private var channelCount = 1

    interface PlaybackListener {
        fun onPlaybackStarted()
        fun onPlaybackCompleted()
        fun onPlaybackError(error: String)
    }

    private var listener: PlaybackListener? = null

    fun setListener(listener: PlaybackListener) {
        this.listener = listener
    }

    fun initializeStreaming() {
        try {
            Log.d(appTag, "Initializing streaming playback")
            stop()
            audioChunkQueue.clear()
            isStreamComplete = false
            shouldContinuePlaying = true
            isPlaying = false
            Log.i(appTag, "Streaming initialized, ready to receive chunks")
        } catch (e: Exception) {
            Log.e(appTag, "Error initializing streaming: ${e.message}", e)
            listener?.onPlaybackError("Failed to initialize streaming: ${e.message}")
        }
    }

    fun addChunk(chunk: ByteArray) {
        try {
            if (chunk.isEmpty()) return

            audioChunkQueue.offer(chunk)
            Log.v(appTag, "Added chunk: ${chunk.size} bytes (queue size: ${audioChunkQueue.size})")

            if (!isPlaying) {
                startStreamingPlayback()
            }
        } catch (e: Exception) {
            Log.e(appTag, "Error adding chunk: ${e.message}", e)
        }
    }

    private fun startStreamingPlayback() {
        if (isPlaying) return

        Log.i(appTag, "Starting streaming playback")
        isPlaying = true
        shouldContinuePlaying = true

        playbackThread = Thread {
            try {
                playStreamingAudio()
            } catch (e: Exception) {
                Log.e(appTag, "Error in playback thread: ${e.message}", e)
                listener?.onPlaybackError("Playback error: ${e.message}")
            } finally {
                cleanupPlaybackResources()
            }
        }
        playbackThread?.start()
    }

    private fun playStreamingAudio() {
        val firstBytes = readInitialBytes() ?: run {
            Log.d(appTag, "No audio data received, skipping playback")
            listener?.onPlaybackCompleted()
            return
        }

        when (detectContainer(firstBytes)) {
            AudioContainer.WAV -> playWavStreamingAudio(firstBytes)
            AudioContainer.MP3 -> playMp3StreamingAudio(firstBytes)
        }
    }

    fun playFile(audioFile: File) {
        if (!audioFile.exists()) {
            listener?.onPlaybackError("Audio file does not exist: ${audioFile.absolutePath}")
            return
        }
        if (audioFile.length() <= MIN_PLAYABLE_AUDIO_BYTES) {
            listener?.onPlaybackError("Audio file is empty or invalid: ${audioFile.length()} bytes")
            return
        }

        try {
            val audioBytes = audioFile.readBytes()
            Log.i(appTag, "Playing TTS file through AudioTrack path: ${audioFile.name} (${audioBytes.size} bytes)")
            initializeStreaming()
            addChunk(audioBytes)
            finalizeStreaming()
        } catch (e: Exception) {
            Log.e(appTag, "Could not play TTS file: ${e.message}", e)
            listener?.onPlaybackError("Could not play TTS file: ${e.message}")
        }
    }

    private fun readInitialBytes(): ByteArray? {
        val firstChunk = readNextChunk() ?: return null
        if (firstChunk.size >= 12 || isStreamComplete) return firstChunk

        val initialBytes = ByteArrayOutputStream()
        initialBytes.write(firstChunk)
        while (initialBytes.size() < 12 && shouldContinuePlaying) {
            val nextChunk = readNextChunk() ?: break
            initialBytes.write(nextChunk)
        }
        return initialBytes.toByteArray()
    }

    private fun detectContainer(firstChunk: ByteArray): AudioContainer =
        if (looksLikeWav(firstChunk)) AudioContainer.WAV else AudioContainer.MP3

    private fun looksLikeWav(bytes: ByteArray): Boolean =
        bytes.size >= 12 &&
            bytes[0] == 'R'.code.toByte() &&
            bytes[1] == 'I'.code.toByte() &&
            bytes[2] == 'F'.code.toByte() &&
            bytes[3] == 'F'.code.toByte() &&
            bytes[8] == 'W'.code.toByte() &&
            bytes[9] == 'A'.code.toByte() &&
            bytes[10] == 'V'.code.toByte() &&
            bytes[11] == 'E'.code.toByte()

    private fun playWavStreamingAudio(firstChunk: ByteArray) {
        Log.i(appTag, "Playing WAV TTS stream")
        listener?.onPlaybackStarted()

        val headerBuffer = ByteArrayOutputStream()
        var wavInfo: WavInfo? = null
        var pendingChunk: ByteArray? = firstChunk
        var totalPcmBytes = 0L

        while (shouldContinuePlaying && wavInfo == null) {
            val chunk = pendingChunk ?: readNextChunk()
            pendingChunk = null
            if (chunk == null) break

            headerBuffer.write(chunk)
            val buffered = headerBuffer.toByteArray()
            wavInfo = parseWavHeader(buffered)

            if (wavInfo != null) {
                configureAudioTrack(wavInfo)
                if (buffered.size > wavInfo.dataStart) {
                    val pcm = buffered.copyOfRange(wavInfo.dataStart, buffered.size)
                    writePcm(pcm)
                    totalPcmBytes += pcm.size
                }
            }
        }

        if (wavInfo == null) {
            throw IllegalArgumentException("Incomplete or unsupported WAV TTS stream")
        }

        while (shouldContinuePlaying) {
            val chunk = readNextChunk() ?: break
            writePcm(chunk)
            totalPcmBytes += chunk.size
        }

        Thread.sleep(250)
        Log.i(appTag, "WAV playback completed successfully ($totalPcmBytes PCM bytes)")
        listener?.onPlaybackCompleted()
    }

    private fun parseWavHeader(bytes: ByteArray): WavInfo? {
        if (bytes.size < 12) return null
        if (!looksLikeWav(bytes)) {
            throw IllegalArgumentException("Audio stream is not a WAV container")
        }

        var offset = 12
        var fmtSampleRate: Int? = null
        var fmtChannels: Int? = null
        var fmtBitsPerSample: Int? = null

        while (offset + 8 <= bytes.size) {
            val chunkSize = readIntLe(bytes, offset + 4)
            if (chunkSize < 0) throw IllegalArgumentException("Invalid WAV chunk size")
            val dataOffset = offset + 8
            val nextOffset = dataOffset + chunkSize + (chunkSize and 1)
            if (dataOffset + chunkSize > bytes.size) return null

            when (fourCc(bytes, offset)) {
                "fmt " -> {
                    if (chunkSize < 16) throw IllegalArgumentException("Invalid WAV fmt chunk")
                    val audioFormat = readShortLe(bytes, dataOffset)
                    if (audioFormat != 1) throw IllegalArgumentException("Only PCM WAV TTS is supported")
                    fmtChannels = readShortLe(bytes, dataOffset + 2)
                    fmtSampleRate = readIntLe(bytes, dataOffset + 4)
                    fmtBitsPerSample = readShortLe(bytes, dataOffset + 14)
                }
                "data" -> {
                    val channels = fmtChannels ?: return null
                    val rate = fmtSampleRate ?: return null
                    val bits = fmtBitsPerSample ?: return null
                    return WavInfo(
                        sampleRate = rate,
                        channelCount = channels,
                        bitsPerSample = bits,
                        dataStart = dataOffset
                    )
                }
            }

            offset = nextOffset
        }

        return null
    }

    private fun configureAudioTrack(info: WavInfo) {
        val channelConfig = when (info.channelCount) {
            1 -> AudioFormat.CHANNEL_OUT_MONO
            2 -> AudioFormat.CHANNEL_OUT_STEREO
            else -> throw IllegalArgumentException("Unsupported WAV channel count: ${info.channelCount}")
        }
        val encoding = when (info.bitsPerSample) {
            8 -> AudioFormat.ENCODING_PCM_8BIT
            16 -> AudioFormat.ENCODING_PCM_16BIT
            else -> throw IllegalArgumentException("Unsupported WAV bit depth: ${info.bitsPerSample}")
        }
        val minBufferSize = AudioTrack.getMinBufferSize(info.sampleRate, channelConfig, encoding)
            .coerceAtLeast(4096)

        sampleRate = info.sampleRate
        channelCount = info.channelCount

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(info.sampleRate)
                    .setChannelMask(channelConfig)
                    .setEncoding(encoding)
                    .build()
            )
            .setBufferSizeInBytes(minBufferSize * 4)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        audioTrack?.play()
        Log.i(appTag, "AudioTrack initialized for WAV: ${info.sampleRate}Hz, ${info.channelCount}ch, ${info.bitsPerSample}bit")
    }

    private fun writePcm(data: ByteArray) {
        var offset = 0
        while (offset < data.size && shouldContinuePlaying) {
            val written = audioTrack?.write(data, offset, data.size - offset) ?: -1
            if (written < 0) {
                throw IllegalStateException("AudioTrack write failed: $written")
            }
            if (written == 0) {
                Thread.sleep(10)
            } else {
                offset += written
            }
        }
    }

    private fun playMp3StreamingAudio(firstChunk: ByteArray) {
        Log.i(appTag, "Playing MP3 TTS stream")
        mediaCodec = MediaCodec.createDecoderByType("audio/mpeg")
        mediaCodec?.configure(MediaFormat.createAudioFormat("audio/mpeg", sampleRate, channelCount), null, null, 0)
        mediaCodec?.start()
        listener?.onPlaybackStarted()

        val info = MediaCodec.BufferInfo()
        var audioTrackInitialized = false
        var totalChunksProcessed = 0
        var inputEOS = false
        var outputEOS = false
        var pendingChunk: ByteArray? = firstChunk

        while (shouldContinuePlaying && !outputEOS) {
            if (!inputEOS) {
                val inputBufferIndex = mediaCodec?.dequeueInputBuffer(10000) ?: -1

                if (inputBufferIndex >= 0) {
                    val inputBuffer = mediaCodec?.getInputBuffer(inputBufferIndex)
                    if (inputBuffer != null) {
                        inputBuffer.clear()
                        val chunk = pendingChunk ?: readNextChunk()
                        pendingChunk = null

                        if (chunk != null) {
                            inputBuffer.put(chunk)
                            mediaCodec?.queueInputBuffer(inputBufferIndex, 0, chunk.size, 0, 0)
                            totalChunksProcessed++
                        } else if (isStreamComplete) {
                            mediaCodec?.queueInputBuffer(
                                inputBufferIndex,
                                0,
                                0,
                                0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            inputEOS = true
                        }
                    }
                }
            }

            when (val outputBufferIndex = mediaCodec?.dequeueOutputBuffer(info, 10000) ?: -1) {
                MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    val outputFormat = mediaCodec?.outputFormat
                    if (outputFormat != null && !audioTrackInitialized) {
                        sampleRate = outputFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channelCount = outputFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        configureAudioTrack(WavInfo(sampleRate, channelCount, 16, 0))
                        audioTrackInitialized = true
                    }
                }
                in 0..Int.MAX_VALUE -> {
                    val outputBuffer = mediaCodec?.getOutputBuffer(outputBufferIndex)
                    if (outputBuffer != null && info.size > 0 && audioTrackInitialized) {
                        val pcmData = ByteArray(info.size)
                        outputBuffer.get(pcmData)
                        writePcm(pcmData)
                    }

                    mediaCodec?.releaseOutputBuffer(outputBufferIndex, false)

                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                        outputEOS = true
                    }
                }
            }
        }

        Thread.sleep(250)
        Log.i(appTag, "MP3 playback completed successfully (processed $totalChunksProcessed chunks)")
        listener?.onPlaybackCompleted()
    }

    private fun readNextChunk(): ByteArray? {
        while (shouldContinuePlaying) {
            val chunk = audioChunkQueue.poll(
                if (isStreamComplete) 0 else 100,
                TimeUnit.MILLISECONDS
            )
            if (chunk != null) return chunk
            if (isStreamComplete) return null
        }
        return null
    }

    fun finalizeStreaming() {
        try {
            Log.d(appTag, "Finalizing streaming (queue size: ${audioChunkQueue.size})")
            isStreamComplete = true

            if (!isPlaying && audioChunkQueue.isEmpty()) {
                listener?.onPlaybackCompleted()
            } else if (!isPlaying) {
                startStreamingPlayback()
            }
        } catch (e: Exception) {
            Log.e(appTag, "Error finalizing streaming: ${e.message}", e)
        }
    }

    private fun cleanupPlaybackResources() {
        try {
            mediaCodec?.apply {
                try {
                    stop()
                } catch (e: IllegalStateException) {
                    Log.w(appTag, "MediaCodec already stopped: ${e.message}")
                }
                release()
            }
            mediaCodec = null

            cleanupMediaPlayer()

            audioTrack?.apply {
                try {
                    pause()
                    flush()
                    stop()
                } catch (e: IllegalStateException) {
                    Log.w(appTag, "AudioTrack already stopped: ${e.message}")
                }
                release()
            }
            audioTrack = null

            isPlaying = false
            Log.d(appTag, "Playback resources cleaned up")
        } catch (e: Exception) {
            Log.e(appTag, "Error cleaning up playback resources: ${e.message}", e)
        }
    }

    fun stop() {
        try {
            Log.d(appTag, "Stopping playback")
            shouldContinuePlaying = false
            isStreamComplete = true

            val thread = playbackThread
            if (thread != null && thread != Thread.currentThread()) {
                thread.join(2000)
            }
            playbackThread = null

            cleanupPlaybackResources()
            cleanupMediaPlayer()
            audioChunkQueue.clear()
            isPlaying = false
        } catch (e: Exception) {
            Log.e(appTag, "Error stopping playback: ${e.message}", e)
        }
    }

    fun isPlaying(): Boolean = isPlaying

    fun release() {
        stop()
        listener = null
        Log.d(appTag, "StreamingAudioPlayer released")
    }

    private fun fourCc(bytes: ByteArray, offset: Int): String =
        String(bytes, offset, 4, Charsets.US_ASCII)

    private fun readShortLe(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xFF) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 8)

    private fun readIntLe(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xFF) or
            ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
            ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
            ((bytes[offset + 3].toInt() and 0xFF) shl 24)

    private fun cleanupMediaPlayer() {
        mediaPlayer?.apply {
            try {
                if (isPlaying) stop()
            } catch (e: IllegalStateException) {
                Log.w(appTag, "MediaPlayer already stopped: ${e.message}")
            }
            release()
        }
        mediaPlayer = null
        isPlaying = false
    }
}
