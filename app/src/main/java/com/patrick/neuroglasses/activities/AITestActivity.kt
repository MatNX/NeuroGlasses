package com.patrick.neuroglasses.activities

import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.patrick.neuroglasses.R
import com.patrick.neuroglasses.helpers.AICameraHelper
import com.patrick.neuroglasses.helpers.AudioHelper
import com.patrick.neuroglasses.helpers.CustomSceneHelper
import com.patrick.neuroglasses.helpers.OpenAIHelper
import com.patrick.neuroglasses.helpers.StreamingAudioPlayer
import com.rokid.cxr.client.extend.CxrApi
import com.rokid.cxr.client.extend.listeners.AiEventListener
import java.io.File

/**
 * AI Chat Configuration Activity
 *
 * This activity allows users to:
 * - Configure whether to include images in AI chat
 * - Configure whether to use ASR (voice) or predefined instructions
 * - Configure whether to use Text-to-Speech (TTS) for responses
 * - Configure teleprompter settings (chunk size and update interval)
 * - Manage predefined instructions
 * - Process AI requests when the AI key is pressed on glasses
 */
class AITestActivity : AppCompatActivity() {
    private val appTag = "AITestActivity"

    // UI Components
    private lateinit var titleTextView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var includeImageCheckBox: CheckBox
    private lateinit var useAsrCheckBox: CheckBox
    private lateinit var useTtsCheckBox: CheckBox
    private lateinit var chunkSizeEditText: EditText
    private lateinit var newInstructionEditText: EditText
    private lateinit var addInstructionButton: Button
    private lateinit var instructionsListView: ListView
    private lateinit var processingStatusTextView: TextView
    private lateinit var capturedImageView: ImageView
    private lateinit var showResultButton: Button

    // Helpers
    private lateinit var aiCameraHelper: AICameraHelper
    private lateinit var audioHelper: AudioHelper
    private lateinit var openAIHelper: OpenAIHelper
    private lateinit var customSceneHelper: CustomSceneHelper
    private lateinit var streamingAudioPlayer: StreamingAudioPlayer

    // Data
    private val predefinedInstructions = mutableListOf<String>()
    private lateinit var instructionsAdapter: ArrayAdapter<String>

    // State
    private var isAiSceneOpen = false
    private var capturedImage: Bitmap? = null
    private var recordedAudioFile: File? = null
    private var lastResultText: String? = null
    private var imageCaptureRequested = false
    private var imageCaptureFinished = true
    private var pendingInstruction: String? = null
    private var pendingInstructionNeedsImage = false
    private var isEndingConversation = false

    // Streaming state
    private var streamingBuffer = StringBuilder()
    private var currentDisplayedCharCount = 0
    private var isStreaming = false

    private fun visibleScrollWindow(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text
        val start = (text.length - maxChars).coerceAtLeast(0)
        val wordBoundary = text.indexOf(' ', start).takeIf { it in start until text.length } ?: start
        return "…" + text.substring(wordBoundary).trimStart()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_second)

        // Initialize UI components
        initializeViews()

        // Initialize helpers
        initializeHelpers()

        // Setup predefined instructions
        setupInstructions()

        // Setup listeners
        setupListeners()

        updateStatus("Bereit. Drücke die KI-Taste an der Brille.")
    }

    private fun initializeViews() {
        titleTextView = findViewById(R.id.secondTitleTextView)
        statusTextView = findViewById(R.id.statusTextView)
        includeImageCheckBox = findViewById(R.id.includeImageCheckBox)
        useAsrCheckBox = findViewById(R.id.useAsrCheckBox)
        useTtsCheckBox = findViewById(R.id.useTtsCheckBox)
        chunkSizeEditText = findViewById(R.id.chunkSizeEditText)
        newInstructionEditText = findViewById(R.id.newInstructionEditText)
        addInstructionButton = findViewById(R.id.addInstructionButton)
        instructionsListView = findViewById(R.id.instructionsListView)
        processingStatusTextView = findViewById(R.id.processingStatusTextView)
        capturedImageView = findViewById(R.id.capturedImageView)
        showResultButton = findViewById(R.id.showResultButton)
    }

    private fun initializeHelpers() {
        // Initialize AI camera helper
        aiCameraHelper = AICameraHelper(appTag)
        aiCameraHelper.setListener(object : AICameraHelper.AICameraListener {
            override fun onCameraOpened(message: String) {
                Log.d(appTag, "Camera opened: $message")
            }

            override fun onCameraOpenFailed(message: String) {
                Log.e(appTag, "Camera open failed: $message")
            }

            override fun onPhotoStatusUpdate(message: String) {
                Log.d(appTag, "Photo status: $message")
            }

            override fun onPhotoSuccess(bitmap: Bitmap, dataSize: Int, width: Int, height: Int) {
                runOnUiThread {
                    capturedImage = bitmap
                    showProcessingUI(true)
                    capturedImageView.setImageBitmap(bitmap)
                    capturedImageView.visibility = View.VISIBLE
                    updateProcessingStatus("Bild aufgenommen: ${width}x${height}")
                    Log.i(appTag, "Photo captured successfully: ${width}x${height}, $dataSize bytes")
                    imageCaptureFinished = true
                    sendPendingInstructionIfReady()
                }
            }

            override fun onPhotoFailed(message: String) {
                runOnUiThread {
                    updateProcessingStatus("Bildaufnahme fehlgeschlagen: $message")
                    Toast.makeText(this@AITestActivity, "Bildaufnahme fehlgeschlagen", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "Photo capture failed: $message")
                    imageCaptureFinished = true
                    sendPendingInstructionIfReady()
                }
            }
        })

        // Initialize audio helper
        audioHelper = AudioHelper(this, appTag)
        audioHelper.setListener(object : AudioHelper.AudioRecordingListener {
            override fun onAudioStreamStarted(codecType: Int, streamType: String?) {
                Log.d(appTag, "Audio stream started: codec=$codecType, stream=$streamType")
            }

            override fun onAudioDataReceived(chunkSize: Int, totalChunks: Int, totalBytes: Long) {
                runOnUiThread {
                    updateProcessingStatus("Audioaufnahme: ${totalBytes / 1024} KB")
                }
            }

            override fun onAudioRecordingStarted(message: String) {
                Log.i(appTag, "Audio recording started: $message")
            }

            override fun onAudioRecordingFailed(message: String) {
                runOnUiThread {
                    updateProcessingStatus("Audioaufnahme fehlgeschlagen: $message")
                    Toast.makeText(this@AITestActivity, "Audioaufnahme fehlgeschlagen", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "Audioaufnahme fehlgeschlagen: $message")
                }
            }

            override fun onAudioRecordingStopped(savedFilePath: String?) {
                runOnUiThread {
                    if (savedFilePath != null) {
                        recordedAudioFile = File(savedFilePath)
                        updateProcessingStatus("Audio gespeichert: ${recordedAudioFile?.name}")
                        Log.i(appTag, "Audio recording saved: $savedFilePath")

                        // Process the request now that audio is ready
                        processAiRequest()
                    } else {
                        updateProcessingStatus("Kein Audio aufgenommen")
                        Log.w(appTag, "No audio data recorded")
                        // Still process request even without audio
                        processAiRequest()
                    }
                }
            }

            override fun onAudioStatusUpdate(message: String) {
                Log.d(appTag, "Audio status: $message")
            }
        })

        // Set audio stream listener
        audioHelper.setAudioStreamListener(true)

        // Initialize OpenAI helper
        openAIHelper = OpenAIHelper(this, appTag)
        openAIHelper.setListener(object : OpenAIHelper.OpenAIListener {
            override fun onAsrComplete(text: String) {
                runOnUiThread {
                    updateProcessingStatus("Spracherkennung: $text")
                    Log.i(appTag, "ASR completed: $text")
                    // Call OpenAI with ASR text (using streaming)
                    sendToOpenAI(text)
                }
            }

            override fun onAsrFailed(error: String) {
                runOnUiThread {
                    updateProcessingStatus("Spracherkennung fehlgeschlagen: $error")
                    Toast.makeText(this@AITestActivity, "Spracherkennung fehlgeschlagen: $error", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "Spracherkennung fehlgeschlagen: $error")
                }
            }

            override fun onOpenAIStreamingStarted() {
                runOnUiThread {
                    updateProcessingStatus("Antwort-Streaming gestartet…")
                    Log.i(appTag, "OpenAI streaming started")
                    isStreaming = true
                    streamingBuffer.clear()
                    currentDisplayedCharCount = 0

                    // Open the custom view to prepare for streaming display
                    customSceneHelper.displayTextResult("")
                }
            }

            override fun onOpenAIStreamingChunk(chunk: String, isComplete: Boolean) {
                runOnUiThread {
                    if (chunk.isNotEmpty()) {
                        // Add chunk to buffer
                        streamingBuffer.append(chunk)
                        currentDisplayedCharCount += chunk.length

                        Log.d(appTag, "Received chunk: '$chunk' (${chunk.length} chars, total: $currentDisplayedCharCount)")

                        // Get the configured chunk size from UI
                        val maxCharsPerDisplay = chunkSizeEditText.text.toString().toIntOrNull() ?: 350
                        val validatedMaxChars = if (maxCharsPerDisplay > 0) maxCharsPerDisplay else 350

                        val visibleText = visibleScrollWindow(streamingBuffer.toString(), validatedMaxChars)
                        currentDisplayedCharCount = visibleText.length
                        customSceneHelper.updateTextResult(visibleText)
                    }

                    if (isComplete) {
                        Log.i(appTag, "Streaming abgeschlossen")
                        isStreaming = false
                        updateProcessingStatus("Streaming abgeschlossen")
                    }
                }
            }

            override fun onOpenAIResponse(response: String) {
                runOnUiThread {
                    updateProcessingStatus("KI-Antwort erhalten")
                    Log.i(appTag, "OpenAI response: $response")

                    // Store the response
                    lastResultText = response

                    // Check if TTS is enabled and streaming is complete
                    if (useTtsCheckBox.isChecked && !isStreaming) {
                        // Convert response to speech
                        updateProcessingStatus("Wandle in Sprache um…")
                        val audioDir = getExternalFilesDir("tts_audio") ?: filesDir
                        openAIHelper.callTtsAPI(response, audioDir)
                    }
                }
            }

            override fun onAssistantToolCall(toolName: String, arguments: Map<String, String>): String? {
                return when (toolName) {
                    "snap_glasses_photo" -> {
                        runOnUiThread {
                            updateProcessingStatus("Brillenfoto wird aufgenommen…")
                            aiCameraHelper.takePhoto()
                        }
                        "Freihändige Fotoaufnahme mit der Rokid-Brillenkamera gestartet. Verwende das Bild für die nächste KI-Anfrage, sobald es verfügbar ist."
                    }
                    else -> null
                }
            }


            override fun onAssistantExternalAppLaunched(label: String) {
                runOnUiThread {
                    Log.i(appTag, "External app launched ($label); stopping assistant session")
                    stopAssistantSessionForExternalApp(label)
                }
            }

            override fun onOpenAIFailed(error: String) {
                runOnUiThread {
                    updateProcessingStatus("KI-Anfrage fehlgeschlagen: $error")
                    Toast.makeText(this@AITestActivity, "KI-Anfrage fehlgeschlagen: $error", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "KI-Anfrage fehlgeschlagen: $error")
                    isStreaming = false
                }
            }

            override fun onTtsComplete(audioFile: File) {
                runOnUiThread {
                    updateProcessingStatus("Sprachausgabe erzeugt")
                    Log.i(appTag, "TTS complete: ${audioFile.absolutePath}")

                    // Set audio file in custom scene helper for potential later use
                    customSceneHelper.setAudioFile(audioFile)

                    // Note: Audio is already playing via streaming, this is just the complete file notification
                    Log.d(appTag, "Complete TTS audio file saved: ${audioFile.absolutePath}")
                }
            }

            override fun onTtsFailed(error: String) {
                runOnUiThread {
                    updateProcessingStatus("TTS failed: $error")
                    Toast.makeText(this@AITestActivity, "Sprachausgabe fehlgeschlagen: $error", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "TTS failed: $error")
                }
            }

            override fun onTtsStreamingStarted() {
                runOnUiThread {
                    updateProcessingStatus("Audio-Streaming gestartet…")
                    Log.i(appTag, "TTS streaming started")

                    // Initialize streaming audio player
                    streamingAudioPlayer.initializeStreaming()
                }
            }

            override fun onTtsStreamingChunk(audioChunk: ByteArray, isComplete: Boolean) {
                runOnUiThread {
                    if (audioChunk.isNotEmpty()) {
                        // Add chunk to streaming player
                        streamingAudioPlayer.addChunk(audioChunk)
                        Log.v(appTag, "TTS chunk added to player: ${audioChunk.size} bytes")
                    }

                    if (isComplete) {
                        // Finalize streaming
                        streamingAudioPlayer.finalizeStreaming()
                        updateProcessingStatus("Audio-Streaming abgeschlossen")
                        Log.i(appTag, "TTS streaming finalized")
                    }
                }
            }
        })

        // Initialize Custom Scene helper
        customSceneHelper = CustomSceneHelper(this, appTag)
        customSceneHelper.setListener(object : CustomSceneHelper.CustomSceneListener {
            override fun onSceneOpened() {
                runOnUiThread {
                    updateProcessingStatus("Ergebnis auf der Brille angezeigt")
                    Toast.makeText(this@AITestActivity, "Ergebnis auf der Brille angezeigt", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onSceneOpenFailed(errorCode: Int) {
                runOnUiThread {
                    updateProcessingStatus("Anzeige auf der Brille fehlgeschlagen: $errorCode")
                    Toast.makeText(this@AITestActivity, "Anzeige auf der Brille fehlgeschlagen: $errorCode", Toast.LENGTH_LONG).show()
                }
            }

            override fun onSceneClosed() {
                runOnUiThread {
                    Log.d(appTag, "Custom scene closed")
                    endConversation("Custom view closed", closeCustomView = false, finishActivity = true)
                }
            }

            override fun onSceneUpdated() {
                runOnUiThread {
                    Log.d(appTag, "Custom scene updated")
                }
            }
        })
        customSceneHelper.initializeCustomViewListener()

        // Initialize Streaming Audio Player
        streamingAudioPlayer = StreamingAudioPlayer(appTag)
        streamingAudioPlayer.setListener(object : StreamingAudioPlayer.PlaybackListener {
            override fun onPlaybackStarted() {
                runOnUiThread {
                    updateProcessingStatus("Audiowiedergabe gestartet")
                    Log.i(appTag, "Streaming audio playback started")
                }
            }

            override fun onPlaybackCompleted() {
                runOnUiThread {
                    updateProcessingStatus("Audiowiedergabe abgeschlossen")
                    Log.i(appTag, "Streaming audio playback completed")
                }
            }

            override fun onPlaybackError(error: String) {
                runOnUiThread {
                    updateProcessingStatus("Audiowiedergabe-Fehler: $error")
                    Toast.makeText(this@AITestActivity, "Audiowiedergabe-Fehler: $error", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "Streaming audio playback error: $error")
                }
            }
        })
    }

    private fun setupInstructions() {
        // Add some default instructions
        predefinedInstructions.add("Was siehst du auf diesem Bild?")
        predefinedInstructions.add("Beschreibe die Szene")
        predefinedInstructions.add("Erkenne Objekte im Bild")
        predefinedInstructions.add("Übersetze den Text in diesem Bild ins Deutsche")

        // Setup adapter
        instructionsAdapter = ArrayAdapter(
            this,
            android.R.layout.simple_list_item_1,
            predefinedInstructions
        )
        instructionsListView.adapter = instructionsAdapter

        // Handle instruction clicks (delete on long click)
        instructionsListView.setOnItemLongClickListener { _, _, position, _ ->
            showDeleteInstructionDialog(position)
            true
        }
    }

    private fun setupListeners() {
        // Add instruction button
        addInstructionButton.setOnClickListener {
            val newInstruction = newInstructionEditText.text.toString().trim()
            if (newInstruction.isNotEmpty()) {
                predefinedInstructions.add(newInstruction)
                instructionsAdapter.notifyDataSetChanged()
                newInstructionEditText.text.clear()
                Toast.makeText(this, "Anweisung hinzugefügt", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Bitte eine Anweisung eingeben", Toast.LENGTH_SHORT).show()
            }
        }

        // Show result button
        showResultButton.setOnClickListener {
            lastResultText?.let { result ->
                displayResultInCustomUI(result)
            }
        }

        // AI event listener
        CxrApi.getInstance().setAiEventListener(object : AiEventListener {
            override fun onAiKeyDown() {
                runOnUiThread {
                    Log.d(appTag, "AI key pressed - starting request")
                    isAiSceneOpen = true
                    onAiKeyPressed()
                }
            }

            override fun onAiKeyUp() {
                // Not used
            }

            override fun onAiExit() {
                runOnUiThread {
                    Log.d(appTag, "AI scene exited")
                    endConversation("AI scene exited", closeCustomView = false, finishActivity = true)
                }
            }
        })
    }

    /**
     * Called when AI key is pressed on glasses
     * This starts the AI request process
     */
    private fun onAiKeyPressed() {
        updateStatus("KI-Anfrage wird verarbeitet…")
        showProcessingUI(true)

        // Stop any ongoing audio playback from previous request
        streamingAudioPlayer.stop()
        customSceneHelper.stopAudio()

        // Reset state
        capturedImage = null
        recordedAudioFile = null
        pendingInstruction = null
        imageCaptureRequested = includeImageCheckBox.isChecked
        imageCaptureFinished = !imageCaptureRequested
        capturedImageView.visibility = View.GONE

        // Always start audio recording (SDK requirement)
        startAudioRecording()

        // Capture image if configured
        if (includeImageCheckBox.isChecked) {
            updateProcessingStatus("Bild wird aufgenommen…")
            aiCameraHelper.takePhoto()
        } else {
            updateProcessingStatus("Bildaufnahme übersprungen (nicht aktiviert)")
        }
    }

    /**
     * Stop the current assistant turn before handing control to a native Android/Rokid app.
     */
    private fun stopAssistantSessionForExternalApp(label: String) {
        updateStatus("$label geöffnet")
        updateProcessingStatus("Assistent gestoppt – App geöffnet")
        endConversation("$label geöffnet", closeCustomView = true, finishActivity = true)
    }

    /**
     * Start audio recording
     */
    private fun startAudioRecording() {
        if (!audioHelper.isRecording) {
            Log.i(appTag, "Starting audio recording")
            audioHelper.clearAudioCache()
            audioHelper.openAudioRecord(codecType = 1, streamType = "AI_assistant")
            updateProcessingStatus("Recording audio...")
        }
    }

    /**
     * Process the AI request
     * Called after audio recording is stopped
     */
    private fun processAiRequest() {
        val useAsr = useAsrCheckBox.isChecked

        if (useAsr) {
            // Use ASR to get text from audio
            processWithASR()
        } else {
            // Show instruction selection dialog
            showInstructionSelectionDialog()
        }
    }

    /**
     * Process request using ASR
     */
    private fun processWithASR() {
        updateProcessingStatus("Processing voice with ASR...")

        if (recordedAudioFile == null) {
            Toast.makeText(this, "Kein Audio aufgenommen", Toast.LENGTH_SHORT).show()
            updateProcessingStatus("Error: No audio available")
            return
        }

        // Call ASR API using helper
        openAIHelper.callAsrAPI(recordedAudioFile!!)
    }

    /**
     * Show dialog to select predefined instruction
     */
    private fun showInstructionSelectionDialog() {
        if (predefinedInstructions.isEmpty()) {
            Toast.makeText(this, "No predefined instructions. Please add some first.", Toast.LENGTH_LONG).show()
            updateProcessingStatus("Error: No predefined instructions")
            return
        }

        AlertDialog.Builder(this)
            .setTitle("Select Instruction")
            .setItems(predefinedInstructions.toTypedArray()) { _, which ->
                val selectedInstruction = predefinedInstructions[which]
                updateProcessingStatus("Instruction: $selectedInstruction")
                sendToOpenAI(selectedInstruction)
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
                updateProcessingStatus("Cancelled")
                showProcessingUI(false)
            }
            .show()
    }

    /**
     * Send request to OpenAI with instruction and optional image
     */
    private fun sendToOpenAI(instruction: String) {
        pendingInstruction = instruction
        pendingInstructionNeedsImage = false
        sendPendingInstructionIfReady()
    }

    private fun sendPendingInstructionIfReady() {
        val instruction = pendingInstruction ?: return

        if (imageCaptureRequested && !imageCaptureFinished) {
            updateProcessingStatus("Warte auf Bildaufnahme…")
            return
        }

        val shouldSendImage = pendingInstructionNeedsImage || includeImageCheckBox.isChecked
        val hasImage = capturedImage != null && shouldSendImage
        pendingInstruction = null
        pendingInstructionNeedsImage = false
        updateProcessingStatus("Sende an KI${if (hasImage) " mit Bild" else ""}…")

        val imageToSend = if (hasImage) capturedImage else null

        // Call OpenAI API using streaming helper
        openAIHelper.callOpenAIStreaming(instruction, imageToSend)
    }

    private fun endConversation(reason: String, closeCustomView: Boolean, finishActivity: Boolean) {
        if (isEndingConversation) return
        isEndingConversation = true
        isAiSceneOpen = false
        isStreaming = false
        pendingInstruction = null
        pendingInstructionNeedsImage = false
        imageCaptureRequested = false
        imageCaptureFinished = true

        updateStatus("Gespräch beendet")
        updateProcessingStatus("Assistent gestoppt: $reason")
        Toast.makeText(this, "Gespräch beendet", Toast.LENGTH_SHORT).show()

        openAIHelper.cancelActiveRequests()
        if (audioHelper.isRecording) {
            audioHelper.closeAudioRecord("AI_assistant")
        }
        streamingAudioPlayer.stop()
        customSceneHelper.stopAudio()
        if (closeCustomView) {
            customSceneHelper.closeCustomView()
        }
        showProcessingUI(false)
        if (finishActivity) {
            finish()
        }
    }

    /**
     * Display result in custom UI on glasses
     */
    private fun displayResultInCustomUI(resultText: String) {
        Log.i(appTag, "Displaying result in custom UI: $resultText")

        // Store the result
        lastResultText = resultText
        showResultButton.isEnabled = true

        updateProcessingStatus("Displaying result on glasses...")

        // Display the full result text directly
        customSceneHelper.displayTextResult(resultText)
        Log.d(appTag, "Ergebnis auf der Brille angezeigt")
    }

    /**
     * Show/hide processing UI elements
     */
    private fun showProcessingUI(show: Boolean) {
        processingStatusTextView.visibility = if (show) View.VISIBLE else View.GONE
        if (!show) {
            capturedImageView.visibility = View.GONE
        }
    }

    /**
     * Update status text
     */
    private fun updateStatus(message: String) {
        statusTextView.text = message
        Log.d(appTag, "Status: $message")
    }

    /**
     * Update processing status text
     */
    private fun updateProcessingStatus(message: String) {
        processingStatusTextView.text = message
        Log.d(appTag, "Processing: $message")
    }

    /**
     * Show dialog to delete instruction
     */
    private fun showDeleteInstructionDialog(position: Int) {
        AlertDialog.Builder(this)
            .setTitle("Delete Instruction")
            .setMessage("Delete \"${predefinedInstructions[position]}\"?")
            .setPositiveButton("Delete") { _, _ ->
                predefinedInstructions.removeAt(position)
                instructionsAdapter.notifyDataSetChanged()
                Toast.makeText(this, "Instruction deleted", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(appTag, "Activity destroying")

        // Release resources
        aiCameraHelper.release()
        audioHelper.release()
        openAIHelper.release()
        customSceneHelper.release()
        streamingAudioPlayer.release()

        // Remove listeners
        CxrApi.getInstance().setAiEventListener(null)
    }
}
