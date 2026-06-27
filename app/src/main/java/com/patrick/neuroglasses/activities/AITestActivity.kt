package com.patrick.neuroglasses.activities

import android.content.Intent
import android.graphics.Bitmap
import android.app.KeyguardManager
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.View
import android.view.WindowManager
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
import com.patrick.neuroglasses.helpers.Message
import com.patrick.neuroglasses.helpers.OpenAIHelper
import com.patrick.neuroglasses.helpers.RokidHostConnection
import com.patrick.neuroglasses.helpers.StreamingAudioPlayer
import com.patrick.neuroglasses.helpers.SystemTtsPlayer
import com.patrick.neuroglasses.services.RokidConnectionService
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
    private lateinit var systemTtsPlayer: SystemTtsPlayer

    // Data
    private val predefinedInstructions = mutableListOf<String>()
    private lateinit var instructionsAdapter: ArrayAdapter<String>

    // State
    private var isAiSceneOpen = false
    private var capturedImage: Bitmap? = null
    private var recordedAudioFile: File? = null
    private var lastResultText: String? = null
    private var currentRequestUsesAsr = false
    private var isAwaitingPhotoForCurrentRequest = false
    private var pendingInstructionForPhoto: String? = null
    private var currentInstructionText: String = ""
    private var hasOpenedStreamingView = false
    private var isConversationActive = false
    private var isClosingConversation = false
    private var ignoreNextAudioStop = false
    private var audioStartAttempts = 0
    private var isAwaitingAsr = false
    private var asrRequestGeneration = 0
    private val sessionConversationMessages = mutableListOf<Message>()

    // Streaming state
    private var streamingBuffer = StringBuilder()
    private var currentDisplayedCharCount = 0
    private var isStreaming = false
    private var isCustomSceneExitArmed = false
    private var sceneExitArmedAtMs = 0L

    companion object {
        const val EXTRA_START_CONVERSATION = "com.patrick.neuroglasses.START_CONVERSATION"
        @Volatile var isVisible = false

        private const val MAX_AUDIO_START_ATTEMPTS = 6
        private const val AUDIO_START_RETRY_DELAY_MS = 350L
        private const val MAX_SESSION_HISTORY_MESSAGES = 20
        private const val ASR_TIMEOUT_MS = 30_000L
        private const val GLASS_APP_RESUME_EXIT_ARM_MS = 1_000L
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        } else {
            @Suppress("DEPRECATION")
            window.addFlags(
                WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED or
                    WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON
            )
        }
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
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
        handleStartConversationIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleStartConversationIntent(intent)
    }

    override fun onStart() {
        super.onStart()
        isVisible = true
    }

    override fun onStop() {
        isVisible = false
        super.onStop()
    }

    private fun handleStartConversationIntent(intent: Intent?) {
        if (intent?.getBooleanExtra(EXTRA_START_CONVERSATION, false) != true) return
        processingStatusTextView.post {
            if (!isFinishing && !isDestroyed) {
                Log.d(appTag, "Starting conversation from glasses foreground service")
                onAiKeyPressed()
            }
        }
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
                    handlePhotoCaptureFinished(photoCaptured = true)
                }
            }

            override fun onPhotoFailed(message: String) {
                runOnUiThread {
                    updateProcessingStatus("Bildaufnahme fehlgeschlagen: $message")
                    Toast.makeText(this@AITestActivity, "Bildaufnahme fehlgeschlagen", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "Photo capture failed: $message")
                    handlePhotoCaptureFinished(photoCaptured = false)
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
                    Log.e(appTag, "Audioaufnahme fehlgeschlagen: $message")
                    if (!isConversationActive || isClosingConversation) {
                        Toast.makeText(this@AITestActivity, "Audioaufnahme fehlgeschlagen", Toast.LENGTH_SHORT).show()
                    }
                }
            }

            override fun onAudioRecordingStopped(savedFilePath: String?) {
                runOnUiThread {
                    if (ignoreNextAudioStop || isClosingConversation) {
                        ignoreNextAudioStop = false
                        Log.d(appTag, "Ignoring audio stop during conversation close")
                        return@runOnUiThread
                    }

                    if (audioHelper.lastStopReason == AudioHelper.StopReason.NO_SPEECH_TIMEOUT) {
                        Log.i(appTag, "Conversation timed out without speech")
                        closeConversation("Nach 15 Sekunden ohne Sprache beendet.")
                        return@runOnUiThread
                    }

                    if (savedFilePath != null) {
                        recordedAudioFile = File(savedFilePath)
                        updateProcessingStatus("Audio gespeichert: ${recordedAudioFile?.name}")
                        Log.i(appTag, "Audio recording saved: $savedFilePath")

                        // Process the request now that audio is ready
                        processAiRequest()
                    } else {
                        updateProcessingStatus("Kein Audio aufgenommen")
                        Log.w(appTag, "No audio data recorded")
                        if (isConversationActive) {
                            restartConversationListening("Ich habe nichts gehört. Sprich bitte nochmal.")
                        } else {
                            processAiRequest()
                        }
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
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    if (!isAwaitingAsr) {
                        Log.d(appTag, "Ignoring stale ASR completion: $text")
                        return@runOnUiThread
                    }
                    isAwaitingAsr = false
                    updateProcessingStatus("Spracherkennung: $text")
                    customSceneHelper.updateTextResult("Verstanden: $text")
                    Log.i(appTag, "ASR completed: $text")
                    handleInstructionReady(text)
                }
            }

            override fun onAsrFailed(error: String) {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    if (!isAwaitingAsr) {
                        Log.d(appTag, "Ignoring stale ASR failure: $error")
                        return@runOnUiThread
                    }
                    isAwaitingAsr = false
                    updateProcessingStatus("Spracherkennung fehlgeschlagen: $error")
                    customSceneHelper.updateTextResult("Spracherkennung fehlgeschlagen.\nDu kannst jetzt sprechen.")
                    Log.e(appTag, "Spracherkennung fehlgeschlagen: $error")
                    restartConversationListening("Ich habe das nicht verstanden. Du kannst jetzt sprechen.")
                }
            }

            override fun onOpenAIStreamingStarted() {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    updateProcessingStatus("Antwort-Streaming gestartet…")
                    Log.i(appTag, "OpenAI streaming started")
                    isStreaming = true
                    streamingBuffer.clear()
                    currentDisplayedCharCount = 0
                    if (hasOpenedStreamingView) {
                        customSceneHelper.updateTextResult("Denke…")
                    } else {
                        hasOpenedStreamingView = true
                        customSceneHelper.displayTextResult("Denke…")
                    }
                }
            }

            override fun onOpenAIStreamingChunk(chunk: String, isComplete: Boolean) {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    if (chunk.isNotEmpty()) {
                        // Add chunk to buffer
                        streamingBuffer.append(chunk)
                        currentDisplayedCharCount += chunk.length

                        Log.d(appTag, "Received chunk: '$chunk' (${chunk.length} chars, total: $currentDisplayedCharCount)")

                        if (!hasOpenedStreamingView) {
                            hasOpenedStreamingView = true
                            customSceneHelper.displayTextResult(streamingBuffer.toString())
                            return@runOnUiThread
                        }

                        // Get the configured chunk size from UI
                        val maxCharsPerDisplay = chunkSizeEditText.text.toString().toIntOrNull() ?: 350
                        val validatedMaxChars = if (maxCharsPerDisplay > 0) maxCharsPerDisplay else 350

                        // Check if we need to clear the display
                        if (currentDisplayedCharCount >= validatedMaxChars) {
                            Log.i(appTag, "Char limit reached ($currentDisplayedCharCount >= $validatedMaxChars), clearing display")
                            // Clear the display and reset counter
                            customSceneHelper.updateTextResult("")
                            // Clear the buffer and reset counter
                            streamingBuffer.clear()
                            streamingBuffer.append(chunk)
                            currentDisplayedCharCount = chunk.length
                            // Update with just the new chunk
                            customSceneHelper.updateTextResult(chunk)
                        } else {
                            // Update the display with accumulated text
                            customSceneHelper.updateTextResult(streamingBuffer.toString())
                        }
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
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread

                    // Store the response
                    lastResultText = response
                    rememberConversationTurn(currentInstructionText, response)

                    // Check if TTS is enabled and streaming is complete
                    if (useTtsCheckBox.isChecked && !isStreaming && shouldSpeakResponse(response)) {
                        // Convert response to speech
                        updateProcessingStatus("Wandle in Sprache um…")
                        Log.d(appTag, "sendTtsContent sent=${RokidHostConnection.sendTtsContent(response)}")
                        if (usesSystemTts()) {
                            systemTtsPlayer.speak(response, SettingsActivity.getTtsVoice(this@AITestActivity))
                        } else {
                            val audioDir = getExternalFilesDir("tts_audio") ?: filesDir
                            openAIHelper.callTtsAPI(response, audioDir, streaming = false)
                        }
                    } else {
                        restartConversationListening()
                    }
                }
            }

            override fun onAssistantToolCall(toolName: String, arguments: Map<String, String>): String? {
                runOnUiThread {
                    if (toolNeedsUnlockedPhone(toolName)) {
                        dismissKeyguardForHandsFreeTool(toolName)
                    }
                    val message = "Führe ${toolDisplayName(toolName)} aus…"
                    updateProcessingStatus(message)
                    if (hasOpenedStreamingView) {
                        customSceneHelper.updateTextResult(message)
                    } else {
                        hasOpenedStreamingView = true
                        customSceneHelper.displayTextResult(message)
                    }
                }
                return when (toolName) {
                    "snap_glasses_photo" -> {
                        if (!includeImageCheckBox.isChecked) {
                            return "Brillenfoto ist deaktiviert. Antworte ohne neues Foto."
                        }
                        runOnUiThread {
                            pendingInstructionForPhoto = currentInstructionText
                            isAwaitingPhotoForCurrentRequest = true
                            capturedImage = null
                            updateProcessingStatus("Brillenfoto wird aufgenommen…")
                            aiCameraHelper.takePhoto()
                        }
                        OpenAIHelper.DEFERRED_PHOTO_TOOL_RESULT
                    }
                    "stop_conversation" -> {
                        runOnUiThread {
                            closeConversation("Konversation beendet.")
                        }
                        OpenAIHelper.LOCAL_TOOL_HANDLED_RESULT
                    }
                    else -> null
                }
            }

            override fun onAssistantToolResult(toolName: String, result: String) {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    if (navigationActuallyStarted(result)) {
                        Log.d(appTag, "Navigation tool result: $result")
                        closeConversation("Navigation gestartet.")
                        return@runOnUiThread
                    }
                    if (isNavigationStartTool(toolName)) {
                        val message = localizeToolResult(result)
                        updateProcessingStatus(message)
                        if (hasOpenedStreamingView) {
                            customSceneHelper.updateTextResult(message)
                        } else {
                            hasOpenedStreamingView = true
                            customSceneHelper.displayTextResult(message)
                        }
                        lastResultText = message
                        rememberConversationTurn(currentInstructionText, message)
                        speakLocalResultOrRestart(message)
                        return@runOnUiThread
                    }
                    if (isNativeRokidTool(toolName)) {
                        Log.d(appTag, "Native Rokid tool result: $result")
                        closeConversation("Native Rokid app opened.", finishAiSession = false)
                        return@runOnUiThread
                    }
                    val message = "${toolDisplayName(toolName)}: ${localizeToolResult(result).take(180)}"
                    updateProcessingStatus(message)
                    if (hasOpenedStreamingView) {
                        customSceneHelper.updateTextResult(message)
                    } else {
                        hasOpenedStreamingView = true
                        customSceneHelper.displayTextResult(message)
                    }
                }
            }

            override fun onOpenAIFailed(error: String) {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    updateProcessingStatus("KI-Anfrage fehlgeschlagen: $error")
                    Toast.makeText(this@AITestActivity, "KI-Anfrage fehlgeschlagen: $error", Toast.LENGTH_SHORT).show()
                    Log.e(appTag, "KI-Anfrage fehlgeschlagen: $error")
                    isStreaming = false
                    if (isConversationActive && !isClosingConversation) {
                        restartConversationListening("KI-Anfrage fehlgeschlagen. Sprich bitte nochmal.")
                    }
                }
            }

            override fun onTtsComplete(audioFile: File) {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    updateProcessingStatus("Sprachausgabe erzeugt (${audioFile.length()} bytes)")
                    Log.i(appTag, "TTS complete: ${audioFile.absolutePath} (${audioFile.length()} bytes)")

                    streamingAudioPlayer.playFile(audioFile)
                }
            }

            override fun onTtsFailed(error: String) {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    updateProcessingStatus("Sprachausgabe nicht verfügbar: $error")
                    customSceneHelper.updateTextResult("Sprachausgabe nicht verfügbar:\n${error.take(140)}")
                    RokidHostConnection.notifyTtsAudioFinished()
                    Log.e(appTag, "TTS failed: $error")
                    if (isConversationActive && !isClosingConversation) {
                        restartConversationListening("Du kannst jetzt sprechen.")
                    }
                }
            }

            override fun onTtsStreamingStarted() {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
                    updateProcessingStatus("Audio-Streaming gestartet…")
                    Log.i(appTag, "TTS streaming started")

                    // Initialize streaming audio player
                    streamingAudioPlayer.initializeStreaming()
                }
            }

            override fun onTtsStreamingChunk(audioChunk: ByteArray, isComplete: Boolean) {
                runOnUiThread {
                    if (!isConversationActive || isClosingConversation) return@runOnUiThread
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
                    if (isConversationActive && !isClosingConversation) {
                        armCustomSceneExit()
                    }
                    updateProcessingStatus("Ergebnis auf der Brille angezeigt")
                    Toast.makeText(this@AITestActivity, "Ergebnis auf der Brille angezeigt", Toast.LENGTH_SHORT).show()
                }
            }

            override fun onSceneOpenFailed(errorCode: Int) {
                runOnUiThread {
                    Log.w(appTag, "Anzeige auf der Brille meldete Fehler: $errorCode")
                }
            }

            override fun onSceneClosed() {
                runOnUiThread {
                    Log.d(appTag, "Custom scene closed")
                    if (!isClosingConversation) {
                        closeConversation("Konversation beendet.")
                    }
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
                handleSpeechPlaybackStarted("Streaming audio playback started")
            }

            override fun onPlaybackCompleted() {
                handleSpeechPlaybackCompleted("Streaming audio playback completed")
            }

            override fun onPlaybackError(error: String) {
                handleSpeechPlaybackError(error, "Streaming audio playback error")
            }
        })

        systemTtsPlayer = SystemTtsPlayer(this, appTag)
        systemTtsPlayer.setListener(object : SystemTtsPlayer.Listener {
            override fun onPlaybackStarted() {
                handleSpeechPlaybackStarted("Android system TTS playback started")
            }

            override fun onPlaybackCompleted() {
                handleSpeechPlaybackCompleted("Android system TTS playback completed")
            }

            override fun onPlaybackError(error: String) {
                handleSpeechPlaybackError(error, "Android system TTS playback error")
            }
        })
    }

    private fun usesSystemTts(): Boolean =
        SettingsActivity.getTtsModel(this).equals(SettingsActivity.DEFAULT_TTS_MODEL, ignoreCase = true)

    private fun speakLocalResultOrRestart(response: String) {
        if (useTtsCheckBox.isChecked && shouldSpeakResponse(response)) {
            RokidHostConnection.sendTtsContent(response)
            if (usesSystemTts()) {
                systemTtsPlayer.speak(response, SettingsActivity.getTtsVoice(this))
            } else {
                val audioDir = getExternalFilesDir("tts_audio") ?: filesDir
                openAIHelper.callTtsAPI(response, audioDir, streaming = false)
            }
        } else {
            restartConversationListening()
        }
    }

    private fun toolNeedsUnlockedPhone(toolName: String): Boolean =
        toolName in setOf(
            "place_phone_call",
            "send_sms",
            "navigation",
            "set_timer",
            "create_reminder",
            "create_calendar_event",
            "open_app",
            "share_text",
            "open_accessibility_settings"
        )

    private fun dismissKeyguardForHandsFreeTool(toolName: String) {
        val keyguardManager = getSystemService(KeyguardManager::class.java) ?: return
        if (!keyguardManager.isKeyguardLocked) return

        Log.d(appTag, "Requesting keyguard dismissal for ${toolDisplayName(toolName)}")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            keyguardManager.requestDismissKeyguard(this, null)
        }
    }

    private fun toolDisplayName(toolName: String): String = when (toolName) {
        "place_phone_call" -> "Anruf"
        "send_sms" -> "SMS"
        "find_contact_phone" -> "Kontakt-Suche"
        "open_rokid_native_app" -> "Rokid-App"
        "navigation" -> "Navigation"
        "start_navigation" -> "Navigation"
        "confirm_navigation_destination" -> "Navigationsziel"
        "stop_navigation" -> "Navigation stoppen"
        "open_rokid_translator" -> "Rokid-Übersetzer"
        "get_gps_location" -> "Standort"
        "get_weather" -> "Wetter"
        "web_search" -> "Suche"
        "set_timer" -> "Timer"
        "create_reminder" -> "Erinnerung"
        "list_reminders" -> "Erinnerungen"
        "create_calendar_event" -> "Kalender"
        "open_app" -> "App"
        "share_text" -> "Teilen"
        "get_battery_status" -> "Akku"
        "memory" -> "Gedächtnis"
        "remember_memory" -> "Erinnerung speichern"
        "search_memory" -> "Gedächtnis"
        "list_memories" -> "Gedächtnis"
        "forget_memory" -> "Erinnerung löschen"
        "saved_place" -> "Orte"
        "save_place" -> "Ort speichern"
        "list_saved_places" -> "Gespeicherte Orte"
        "navigate_saved_place" -> "Ort-Navigation"
        "forget_saved_place" -> "Ort löschen"
        "get_agenda" -> "Agenda"
        "find_nearby_places" -> "Umgebung"
        "get_transit_departures" -> "Abfahrten"
        "open_accessibility_settings" -> "Bedienungshilfen"
        "snap_glasses_photo" -> "Foto"
        "chat" -> "Chat"
        "new_chat" -> "Neuer Chat"
        "list_chats" -> "Chats"
        "switch_chat" -> "Chat wechseln"
        "rename_chat" -> "Chat umbenennen"
        "leave_chat" -> "Chat verlassen"
        "delete_chat" -> "Chat löschen"
        "stop_conversation" -> "Stopp"
        else -> toolName
    }

    private fun localizeToolResult(result: String): String = when {
        result.startsWith("Gestartet:", ignoreCase = true) -> "gestartet"
        result.startsWith("Started ", ignoreCase = true) -> "gestartet"
        result.startsWith("Angefordert:", ignoreCase = true) -> "angefragt"
        result.startsWith("Requested ", ignoreCase = true) -> "angefragt"
        result.startsWith("SMS an", ignoreCase = true) -> "gesendet"
        result.startsWith("SMS sent", ignoreCase = true) -> "gesendet"
        result.startsWith("Gefunden:", ignoreCase = true) -> result
        result.startsWith("Found ", ignoreCase = true) -> result
        result.startsWith("Konnte", ignoreCase = true) -> "fehlgeschlagen: $result"
        result.startsWith("Could not", ignoreCase = true) -> "fehlgeschlagen: $result"
        result.contains("Berechtigung", ignoreCase = true) -> "Berechtigung fehlt: $result"
        result.contains("permission", ignoreCase = true) -> "Berechtigung fehlt: $result"
        else -> result
    }

    private fun isNavigationStartTool(toolName: String): Boolean =
        toolName == "navigation" ||
            toolName == "start_navigation" ||
            toolName == "confirm_navigation_destination" ||
            toolName == "navigate_saved_place"

    private fun navigationActuallyStarted(result: String): Boolean =
        result.startsWith("Navigation gestartet", ignoreCase = true) ||
            result.startsWith("Navigation mode started", ignoreCase = true)

    private fun isNativeRokidTool(toolName: String): Boolean =
        toolName == "open_rokid_native_app" ||
            toolName == "open_rokid_translator"

    private fun rememberConversationTurn(userText: String, assistantText: String) {
        val cleanUserText = userText.trim()
        val cleanAssistantText = assistantText.trim()
        if (cleanUserText.isBlank() || cleanAssistantText.isBlank()) return

        sessionConversationMessages += Message(role = "user", content = cleanUserText)
        sessionConversationMessages += Message(role = "assistant", content = cleanAssistantText)
        while (sessionConversationMessages.size > MAX_SESSION_HISTORY_MESSAGES) {
            sessionConversationMessages.removeAt(0)
        }
    }

    private fun handleSpeechPlaybackStarted(logMessage: String) {
        runOnUiThread {
            updateProcessingStatus("Audiowiedergabe gestartet")
            Log.i(appTag, logMessage)
        }
    }

    private fun handleSpeechPlaybackCompleted(logMessage: String) {
        runOnUiThread {
            updateProcessingStatus("Audiowiedergabe abgeschlossen")
            Log.i(appTag, logMessage)
            RokidHostConnection.notifyTtsAudioFinished()
            if (isConversationActive && !isClosingConversation) {
                restartConversationListening("Du kannst jetzt sprechen.")
            }
        }
    }

    private fun handleSpeechPlaybackError(error: String, logPrefix: String) {
        runOnUiThread {
            updateProcessingStatus("Audiowiedergabe-Fehler: $error")
            customSceneHelper.updateTextResult("Audiowiedergabe fehlgeschlagen:\n${error.take(140)}")
            RokidHostConnection.notifyTtsAudioFinished()
            Log.e(appTag, "$logPrefix: $error")
            if (isConversationActive && !isClosingConversation) {
                restartConversationListening("Audiowiedergabe fehlgeschlagen. Du kannst jetzt sprechen.")
            } else {
                Toast.makeText(this@AITestActivity, "Audiowiedergabe-Fehler: $error", Toast.LENGTH_SHORT).show()
            }
        }
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
        RokidConnectionService.start(this)
        RokidHostConnection.setInterruptAiWake(true)
        RokidHostConnection.connect(this)
        RokidHostConnection.setInterruptAiWake(true)
        RokidHostConnection.setAiEventListener(object : RokidHostConnection.AiEventListener {
            override fun onAiKeyDown() {
                runOnUiThread {
                    Log.d(
                        appTag,
                        "AI key down delivered: active=$isConversationActive " +
                            "closing=$isClosingConversation sceneArmed=$isCustomSceneExitArmed"
                    )
                    RokidHostConnection.setInterruptAiWake(true)
                    RokidHostConnection.interruptAiWakeNow()
                    onAiKeyPressed()
                }
            }

            override fun onAiKeyUp() {
                // Not used
            }

            override fun onAiExit() {
                runOnUiThread {
                    Log.d(appTag, "AI scene exited")
                    closeConversation("Konversation beendet.")
                }
            }

            override fun onGlassAppResumeChanged(firstPackage: String?, secondPackage: String?) {
                runOnUiThread {
                    handleGlassAppResumeChange(firstPackage, secondPackage)
                }
            }
        })
    }

    /**
     * Called when AI key is pressed on glasses
     * This starts the AI request process
     */
    private fun onAiKeyPressed() {
        if (isConversationActive && !isClosingConversation) {
            val armedForMs = if (sceneExitArmedAtMs > 0L) {
                SystemClock.elapsedRealtime() - sceneExitArmedAtMs
            } else {
                0L
            }
            Log.d(appTag, "AI key pressed while conversation active; ending conversation. armedForMs=$armedForMs")
            closeConversation("Konversation beendet.")
            return
        }

        Log.d(appTag, "AI key pressed while conversation inactive; starting conversation")
        isConversationActive = true
        isClosingConversation = false
        disarmCustomSceneExit()
        ignoreNextAudioStop = false
        audioStartAttempts = 0
        sessionConversationMessages.clear()
        isAiSceneOpen = true
        updateStatus("Neuro hört zu…")
        showProcessingUI(true)
        currentRequestUsesAsr = useAsrCheckBox.isChecked
        Log.d(appTag, "setCommunicationDevice sent=${RokidHostConnection.setCommunicationDevice()}")

        // Stop any ongoing audio playback from previous request
        streamingAudioPlayer.stop()
        customSceneHelper.stopAudio()
        customSceneHelper.closeCustomView()

        resetCurrentTurn()

        if (currentRequestUsesAsr) {
            openListeningView()
            startAudioRecording()
        } else {
            updateProcessingStatus("Spracheingabe deaktiviert")
            showInstructionSelectionDialog()
        }
    }

    /**
     * Start audio recording
     */
    private fun startAudioRecording() {
        if (!audioHelper.isRecording) {
            Log.i(appTag, "Starting audio recording")
            audioHelper.clearAudioCache()
            val started = audioHelper.openAudioRecord(codecType = 1, streamType = "AI_assistant", autoStopOnSilence = true)
            if (started) {
                audioStartAttempts = 0
                val readyMessage = "Du kannst jetzt sprechen."
                updateProcessingStatus(readyMessage)
                customSceneHelper.updateTextResult(readyMessage)
            } else if (isConversationActive) {
                audioStartAttempts++
                if (audioStartAttempts < MAX_AUDIO_START_ATTEMPTS) {
                    updateProcessingStatus("Audio wird verbunden…")
                    processingStatusTextView.postDelayed({
                        if (isConversationActive && !isClosingConversation && !audioHelper.isRecording) {
                            startAudioRecording()
                        }
                    }, AUDIO_START_RETRY_DELAY_MS)
                } else {
                    audioStartAttempts = 0
                    closeConversation("Audioaufnahme konnte nicht gestartet werden.")
                }
            }
        }
    }

    private fun resetCurrentTurn() {
        isAwaitingPhotoForCurrentRequest = false
        pendingInstructionForPhoto = null
        currentInstructionText = ""
        hasOpenedStreamingView = false
        capturedImage = null
        recordedAudioFile = null
        capturedImageView.visibility = View.GONE
    }

    private fun openListeningView() {
        hasOpenedStreamingView = true
        customSceneHelper.displayTextResult("Ich höre zu. Sprich jetzt.")
    }

    private fun restartConversationListening(message: String = "Du kannst jetzt sprechen.") {
        if (!isConversationActive || isClosingConversation) return
        if (!useAsrCheckBox.isChecked) {
            closeConversation("Konversation beendet. Spracheingabe ist deaktiviert.")
            return
        }
        if (audioHelper.isRecording) return

        currentRequestUsesAsr = true
        resetCurrentTurn()
        audioStartAttempts = 0
        updateStatus("Konversation läuft")
        updateProcessingStatus(message)
        if (hasOpenedStreamingView) {
            customSceneHelper.updateTextResult(message)
        } else {
            openListeningView()
        }
        startAudioRecording()
    }

    private fun closeConversation(message: String, finishAiSession: Boolean = true) {
        if (isClosingConversation) return

        Log.i(appTag, "Closing conversation: $message")
        isClosingConversation = true
        isConversationActive = false
        isAiSceneOpen = false
        disarmCustomSceneExit()
        isAwaitingPhotoForCurrentRequest = false
        pendingInstructionForPhoto = null
        currentInstructionText = ""
        hasOpenedStreamingView = false
        isStreaming = false
        audioStartAttempts = 0
        isAwaitingAsr = false
        asrRequestGeneration++
        sessionConversationMessages.clear()

        openAIHelper.cancelAsrRequest("conversation closed")
        streamingAudioPlayer.stop()
        systemTtsPlayer.stop()
        customSceneHelper.stopAudio()
        RokidHostConnection.notifyTtsAudioFinished()

        ignoreNextAudioStop = true
        audioHelper.abortRecordingForSessionClose()

        customSceneHelper.closeCustomView()
        Log.d(appTag, "clearCommunicationDevice sent=${RokidHostConnection.clearCommunicationDevice()}")
        if (finishAiSession) {
            RokidHostConnection.finishAiSession()
        }
        showProcessingUI(false)
        updateStatus(message)
        isClosingConversation = false
    }

    private fun armCustomSceneExit() {
        isCustomSceneExitArmed = true
        sceneExitArmedAtMs = SystemClock.elapsedRealtime()
    }

    private fun disarmCustomSceneExit() {
        isCustomSceneExitArmed = false
        sceneExitArmedAtMs = 0L
    }

    private fun handleGlassAppResumeChange(firstPackage: String?, secondPackage: String?) {
        if (!isConversationActive || isClosingConversation) return
        val armedForMs = if (sceneExitArmedAtMs > 0L) {
            SystemClock.elapsedRealtime() - sceneExitArmedAtMs
        } else {
            0L
        }
        if (!isCustomSceneExitArmed || armedForMs < GLASS_APP_RESUME_EXIT_ARM_MS) {
            Log.d(
                appTag,
                "Ignoring early glass app resume change: first=$firstPackage second=$secondPackage " +
                    "armed=$isCustomSceneExitArmed armedForMs=$armedForMs"
            )
            return
        }

        val customViewOpen = customSceneHelper.isCustomViewOpen()
        val audioStreaming = RokidHostConnection.isAudioStreaming()
        Log.d(
            appTag,
            "Glass app resume changed while custom scene is active: first=$firstPackage " +
                "second=$secondPackage customViewOpen=$customViewOpen audioStreaming=$audioStreaming"
        )
        closeConversation("Konversation beendet.")
    }

    private fun handlePhotoCaptureFinished(photoCaptured: Boolean) {
        if (!isAwaitingPhotoForCurrentRequest) return

        isAwaitingPhotoForCurrentRequest = false
        val pendingInstruction = pendingInstructionForPhoto
        pendingInstructionForPhoto = null

        if (pendingInstruction != null) {
            val imageToSend = if (photoCaptured) capturedImage else null
            if (!photoCaptured) capturedImage = null
            sendToOpenAI(
                instruction = pendingInstruction,
                image = imageToSend,
                allowPhotoTool = false
            )
        }
    }

    private fun handleInstructionReady(instruction: String) {
        val cleanInstruction = instruction.trim()
        if (cleanInstruction.isBlank()) {
            restartConversationListening("Ich habe nichts verstanden. Sprich bitte nochmal.")
            return
        }

        sendToOpenAI(cleanInstruction)
    }

    /**
     * Process the AI request
     * Called after audio recording is stopped
     */
    private fun processAiRequest() {
        if (currentRequestUsesAsr) {
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
        val message = "Spracherkennung läuft…"
        updateProcessingStatus(message)
        if (hasOpenedStreamingView) {
            customSceneHelper.updateTextResult(message)
        } else {
            hasOpenedStreamingView = true
            customSceneHelper.displayTextResult(message)
        }

        if (recordedAudioFile == null) {
            updateProcessingStatus("Error: No audio available")
            if (isConversationActive) {
                restartConversationListening("Ich habe nichts gehört. Sprich bitte nochmal.")
            } else {
                Toast.makeText(this, "Kein Audio aufgenommen", Toast.LENGTH_SHORT).show()
            }
            return
        }

        val audioFile = recordedAudioFile!!
        if (audioFile.length() <= 44L) {
            updateProcessingStatus("Kein verwertbares Audio aufgenommen")
            restartConversationListening("Ich habe nichts gehört. Du kannst jetzt sprechen.")
            return
        }

        isAwaitingAsr = true
        val requestGeneration = ++asrRequestGeneration
        processingStatusTextView.postDelayed({
            if (
                isConversationActive &&
                !isClosingConversation &&
                isAwaitingAsr &&
                requestGeneration == asrRequestGeneration
            ) {
                Log.w(appTag, "ASR timed out after ${ASR_TIMEOUT_MS}ms")
                isAwaitingAsr = false
                openAIHelper.cancelAsrRequest("ASR timed out")
                restartConversationListening("Spracherkennung dauert zu lange. Du kannst jetzt sprechen.")
            }
        }, ASR_TIMEOUT_MS)

        // Call ASR API using helper
        openAIHelper.callAsrAPI(audioFile)
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
                handleInstructionReady(selectedInstruction)
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
                updateProcessingStatus("Cancelled")
                closeConversation("Konversation abgebrochen.")
            }
            .show()
    }

    /**
     * Send request to OpenAI with instruction and optional image
     */
    private fun sendToOpenAI(
        instruction: String,
        image: Bitmap? = null,
        allowPhotoTool: Boolean = includeImageCheckBox.isChecked
    ) {
        if (!isConversationActive || isClosingConversation) return
        updateProcessingStatus("Sending to AI...")
        currentInstructionText = instruction

        // Call OpenAI API using streaming helper
        openAIHelper.callOpenAIStreaming(
            instruction = instruction,
            image = image,
            allowGlassesPhotoTool = allowPhotoTool && image == null,
            includePersistedHistory = false,
            persistConversation = false,
            conversationHistory = sessionConversationMessages.toList()
        )
    }

    private fun shouldSpeakResponse(response: String): Boolean {
        if (response.isBlank()) return false
        if (response.contains("TTS API call failed", ignoreCase = true)) return false
        return true
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

        disarmCustomSceneExit()

        // Release resources
        aiCameraHelper.release()
        audioHelper.release()
        openAIHelper.release()
        customSceneHelper.release()
        streamingAudioPlayer.release()
        systemTtsPlayer.release()

        // Remove listeners
        RokidHostConnection.setAiEventListener(null)
    }
}
