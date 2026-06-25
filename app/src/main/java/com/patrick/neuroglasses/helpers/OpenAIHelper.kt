package com.patrick.neuroglasses.helpers

import android.Manifest
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.AlarmClock
import android.provider.CalendarContract
import android.provider.Settings
import android.provider.ContactsContract
import android.telephony.SmsManager
import android.location.LocationManager
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Base64
import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.patrick.neuroglasses.activities.SettingsActivity
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.util.Collections
import java.net.URLEncoder
import java.util.Locale
import java.util.UUID

/**
 * Data classes for OpenAI API request/response
 */
data class OpenAIRequest(
    val model: String,
    val messages: List<Message>,
    @SerializedName("max_tokens")
    val maxTokens: Int,
    val stream: Boolean = false,
    val tools: List<AssistantTool>? = null,
    @SerializedName("tool_choice")
    val toolChoice: String? = null
)

data class Message(
    val role: String,
    val content: Any? = null,
    @SerializedName("tool_calls")
    val toolCalls: List<ToolCall>? = null,
    @SerializedName("tool_call_id")
    val toolCallId: String? = null
)

data class AssistantTool(
    val type: String = "function",
    val function: ToolFunction
)

data class ToolFunction(
    val name: String,
    val description: String,
    val parameters: Map<String, Any>
)

data class ToolCall(
    val id: String,
    val type: String,
    val function: ToolCallFunction
)

data class ToolCallFunction(
    val name: String,
    val arguments: String
)

data class Content(
    val type: String,
    val text: String? = null,
    @SerializedName("image_url")
    val imageUrl: ImageUrl? = null
)

data class ImageUrl(
    val url: String
)

/**
 * Data classes for streaming responses
 */
data class ChatCompletionResponse(
    val choices: List<ChatCompletionChoice>
)

data class ChatCompletionChoice(
    val message: Message,
    @SerializedName("finish_reason")
    val finishReason: String?
)

data class StreamingResponse(
    val id: String,
    val choices: List<StreamingChoice>
)

data class StreamingChoice(
    val delta: Delta,
    @SerializedName("finish_reason")
    val finishReason: String?
)

data class Delta(
    val role: String?,
    val content: String?
)

/**
 * Data class for ASR API response
 */
data class AsrResponse(
    val text: String
)

/**
 * Data class for TTS API request
 */
data class TtsRequest(
    val model: String,
    val input: String,
    val voice: String
)

data class PersistedChatMessage(
    val role: String,
    val text: String,
    val timestamp: Long = System.currentTimeMillis()
)

data class PersistedChatSession(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val messages: List<PersistedChatMessage> = emptyList()
)

/**
 * Groq AI Helper
 * Handles AI-related API calls including ASR (speech-to-text) and OpenAI chat completion
 */
class OpenAIHelper(private val context: Context, private val appTag: String = "OpenAIHelper") {

    private val gson = Gson()
    private val chatPrefs by lazy { context.getSharedPreferences("assistant_chats", Context.MODE_PRIVATE) }
    private val activeCalls = Collections.synchronizedSet(mutableSetOf<Call>())


    // Get configuration from SharedPreferences
    private fun getApiBaseUrl(): String = SettingsActivity.getApiBaseUrl(context).trimEnd('/')
    private fun getApiToken(): String = SettingsActivity.getApiToken(context)
    private fun getApiTimeout(): Int = SettingsActivity.getApiTimeout(context)
    private fun getSystemPrompt(): String = SettingsActivity.getSystemPrompt(context)
    private fun getVlmModel(): String = SettingsActivity.getVlmModel(context)
    private fun getVlmMaxTokens(): Int = SettingsActivity.getVlmMaxTokens(context)
    private fun getAsrModel(): String = SettingsActivity.getAsrModel(context)
    private fun getTtsModel(): String = SettingsActivity.getTtsModel(context)
    private fun getTtsVoice(): String = SettingsActivity.getTtsVoice(context)
    private fun getMapTilerApiKey(): String = SettingsActivity.getMapTilerApiKey(context)

    // Build OkHttpClient with configurable timeout
    private fun getClient(isStreaming: Boolean = false): OkHttpClient {
        val timeoutSeconds = getApiTimeout().toLong()
        return OkHttpClient.Builder()
            .connectTimeout(timeoutSeconds, TimeUnit.SECONDS)
            // For streaming, use a longer read timeout to allow for slower chunk delivery
            // but still respect the configured timeout as a baseline
            .readTimeout(if (isStreaming) timeoutSeconds * 2 else timeoutSeconds, TimeUnit.SECONDS)
            .writeTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .build()
    }

    private fun getChatApiUrl(): String = "${getApiBaseUrl()}/chat/completions"
    private fun getAsrApiUrl(): String = "${getApiBaseUrl()}/audio/transcriptions"
    private fun getTtsApiUrl(): String = "${getApiBaseUrl()}/audio/speech"

    fun cancelActiveRequests() {
        synchronized(activeCalls) {
            activeCalls.forEach { call -> runCatching { call.cancel() } }
            activeCalls.clear()
        }
        shouldStopAssistantTurn = true
        Log.i(appTag, "Cancelled active AI network requests")
    }

    private fun executeTracked(request: Request, isStreaming: Boolean = false): Response {
        val call = getClient(isStreaming = isStreaming).newCall(request)
        activeCalls.add(call)
        return call.execute()
    }


    private fun getChatSessions(): MutableList<PersistedChatSession> {
        val json = chatPrefs.getString("sessions", null) ?: return mutableListOf()
        return runCatching {
            gson.fromJson(json, Array<PersistedChatSession>::class.java).toMutableList()
        }.getOrElse {
            Log.w(appTag, "Could not parse persisted chats", it)
            mutableListOf()
        }
    }

    private fun saveChatSessions(sessions: List<PersistedChatSession>) {
        chatPrefs.edit().putString("sessions", gson.toJson(sessions)).apply()
    }

    private fun activeChatId(): String {
        val sessions = getChatSessions()
        val stored = chatPrefs.getString("active_chat_id", null)
        if (stored != null && sessions.any { it.id == stored }) return stored
        val session = sessions.firstOrNull() ?: createChatSession("Chat 1")
        chatPrefs.edit().putString("active_chat_id", session.id).apply()
        return session.id
    }

    private fun createChatSession(title: String? = null): PersistedChatSession {
        val sessions = getChatSessions()
        val now = System.currentTimeMillis()
        val session = PersistedChatSession(
            id = UUID.randomUUID().toString(),
            title = title?.takeIf { it.isNotBlank() } ?: "Chat ${sessions.size + 1}",
            createdAt = now,
            updatedAt = now
        )
        sessions.add(0, session)
        saveChatSessions(sessions)
        chatPrefs.edit().putString("active_chat_id", session.id).apply()
        return session
    }

    private fun updateChatSession(chatId: String, transform: (PersistedChatSession) -> PersistedChatSession): PersistedChatSession? {
        val sessions = getChatSessions()
        val index = sessions.indexOfFirst { it.id == chatId }
        if (index == -1) return null
        val updated = transform(sessions[index]).copy(updatedAt = System.currentTimeMillis())
        sessions[index] = updated
        saveChatSessions(sessions.sortedByDescending { it.updatedAt })
        return updated
    }

    private fun appendChatMessages(chatId: String, messages: List<PersistedChatMessage>) {
        if (messages.isEmpty()) return
        updateChatSession(chatId) { session ->
            val title = if (session.messages.isEmpty() && session.title.startsWith("Chat ")) {
                messages.firstOrNull { it.role == "user" }?.text?.take(40)?.ifBlank { session.title } ?: session.title
            } else session.title
            session.copy(title = title, messages = (session.messages + messages).takeLast(40))
        }
    }

    private fun activeChatHistoryMessages(): List<Message> {
        val session = getChatSessions().firstOrNull { it.id == activeChatId() } ?: return emptyList()
        return session.messages.takeLast(20).map { Message(role = it.role, content = it.text) }
    }

    private fun switchChat(chatId: String?, title: String?): String {
        val sessions = getChatSessions()
        val session = when {
            !chatId.isNullOrBlank() -> sessions.firstOrNull { it.id == chatId }
            !title.isNullOrBlank() -> sessions.firstOrNull { it.title.equals(title, ignoreCase = true) }
            else -> null
        } ?: return "Chat nicht gefunden. Nutze list_chats, um verfügbare Chat-IDs und Titel zu sehen."
        chatPrefs.edit().putString("active_chat_id", session.id).apply()
        return "Zu Chat '${session.title}' (${session.id}) gewechselt."
    }

    private fun listChats(): String {
        val active = activeChatId()
        val sessions = getChatSessions()
        if (sessions.isEmpty()) return "Es gibt noch keine Chats."
        return sessions.joinToString("\n") { session ->
            val marker = if (session.id == active) "*" else "-"
            "$marker ${session.title} [${session.id}] (${session.messages.size} Nachrichten)"
        }
    }

    private fun renameChat(chatId: String?, title: String): String {
        if (title.isBlank()) return "Ein neuer Chat-Titel ist erforderlich."
        val targetId = chatId?.takeIf { it.isNotBlank() } ?: activeChatId()
        val updated = updateChatSession(targetId) { it.copy(title = title) }
        return if (updated == null) "Chat nicht gefunden." else "Chat in '${updated.title}' umbenannt."
    }

    /**
     * Listener interface for AI API callbacks
     */
    interface OpenAIListener {
        /**
         * Called when ASR processing is complete
         * @param text The recognized text from audio
         */
        fun onAsrComplete(text: String)

        /**
         * Called when ASR processing fails
         * @param error Error message
         */
        fun onAsrFailed(error: String)

        /**
         * Called when OpenAI API response is received (non-streaming)
         * @param response The AI-generated response text
         */
        fun onOpenAIResponse(response: String)

        /**
         * Called when OpenAI streaming starts
         */
        fun onOpenAIStreamingStarted() {}

        /**
         * Called when a streaming chunk is received
         * @param chunk The text chunk received
         * @param isComplete Whether this is the final chunk
         */
        fun onOpenAIStreamingChunk(chunk: String, isComplete: Boolean) {}

        /**
         * Gives the activity first chance to handle glasses-specific tool calls such as snapshots.
         * Return null to let OpenAIHelper handle the tool locally.
         */
        fun onAssistantToolCall(toolName: String, arguments: Map<String, String>): String? = null

        /**
         * Called when a tool successfully opens another Android app or system screen.
         * The current assistant turn should stop immediately so it does not reopen the
         * glasses chat UI or keep audio resources alive over the launched app.
         */
        fun onAssistantExternalAppLaunched(label: String) {}

        /**
         * Called when OpenAI API call fails
         * @param error Error message
         */
        fun onOpenAIFailed(error: String)

        /**
         * Called when TTS audio file is ready
         * @param audioFile The generated audio file
         */
        fun onTtsComplete(audioFile: File)

        /**
         * Called when TTS processing fails
         * @param error Error message
         */
        fun onTtsFailed(error: String)

        /**
         * Called when TTS streaming starts
         */
        fun onTtsStreamingStarted() {}

        /**
         * Called when a TTS audio chunk is received
         * @param audioChunk The audio data chunk
         * @param isComplete Whether this is the final chunk
         */
        fun onTtsStreamingChunk(audioChunk: ByteArray, isComplete: Boolean) {}
    }

    private var listener: OpenAIListener? = null

    /**
     * Set the listener for AI API callbacks
     */
    fun setListener(listener: OpenAIListener) {
        this.listener = listener
    }

    /**
     * Call ASR API to convert audio to text
     * @param audioFile The audio file to process
     */
    fun callAsrAPI(audioFile: File) {
        Log.d(appTag, "ASR API called with file: ${audioFile.name}")

        if (!audioFile.exists()) {
            Log.e(appTag, "Audio file does not exist: ${audioFile.path}")
            listener?.onAsrFailed("Audio file not found")
            return
        }

        Thread {
            try {
                // Create multipart form data request
                val requestBody = MultipartBody.Builder()
                    .setType(MultipartBody.FORM)
                    .addFormDataPart("model", getAsrModel())
                    .addFormDataPart("language", "de")
                    .addFormDataPart("prompt", "Dies ist eine deutschsprachige Sprachsteuerung für eine AR-Brille in Österreich.")
                    .addFormDataPart(
                        "file",
                        audioFile.name,
                        audioFile.asRequestBody("audio/wav".toMediaType())
                    )
                    .build()

                // Create HTTP request
                val request = Request.Builder()
                    .url(getAsrApiUrl())
                    .addHeader("Authorization", "Bearer ${getApiToken()}")
                    .post(requestBody)
                    .build()

                Log.d(appTag, "Sending ASR request for file: ${audioFile.name} (${audioFile.length()} bytes)")

                // Execute the request
                executeTracked(request).use { response ->
                    if (!response.isSuccessful) {
                        val errorBody = response.body?.string() ?: "Unknown error"
                        Log.e(appTag, "ASR API call failed: ${response.code} - $errorBody")
                        listener?.onAsrFailed("ASR API call failed: ${response.code}")
                        return@Thread
                    }

                    val responseBody = response.body?.string()
                    Log.d(appTag, "ASR Response: $responseBody")

                    if (responseBody != null) {
                        val asrResponse = gson.fromJson(responseBody, AsrResponse::class.java)
                        val recognizedText = asrResponse.text

                        if (recognizedText.isNotEmpty()) {
                            Log.d(appTag, "ASR recognized text: $recognizedText")
                            listener?.onAsrComplete(recognizedText)
                        } else {
                            Log.e(appTag, "ASR returned empty text")
                            listener?.onAsrFailed("No text recognized")
                        }
                    } else {
                        Log.e(appTag, "Empty ASR response body")
                        listener?.onAsrFailed("Empty response")
                    }
                }
            } catch (e: IOException) {
                Log.e(appTag, "Network error during ASR: ${e.message}", e)
                listener?.onAsrFailed("Network error: ${e.message}")
            } catch (e: Exception) {
                Log.e(appTag, "Error calling ASR API: ${e.message}", e)
                listener?.onAsrFailed("Error: ${e.message}")
            }
        }.start()
    }


    private fun assistantTools(): List<AssistantTool> {
        fun objectSchema(required: List<String>, properties: Map<String, Map<String, Any>>): Map<String, Any> = mapOf(
            "type" to "object",
            "required" to required,
            "properties" to properties,
            "additionalProperties" to false
        )

        fun stringProp(description: String): Map<String, Any> = mapOf("type" to "string", "description" to description)

        return listOf(
            AssistantTool(function = ToolFunction(
                name = "place_phone_call",
                description = "Deutschsprachig: Rufe freihändig eine Telefonnummer oder einen Kontakt an, wenn die CALL_PHONE-Berechtigung vorhanden ist; sonst erkläre kurz, dass die Berechtigung fehlt.",
                parameters = objectSchema(listOf("recipient"), mapOf("recipient" to stringProp("Telefonnummer oder Kontaktname.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "send_sms",
                description = "Deutschsprachig: Sende freihändig eine SMS an eine Telefonnummer oder einen Kontakt, wenn die SEND_SMS-Berechtigung vorhanden ist.",
                parameters = objectSchema(listOf("recipient", "message"), mapOf(
                    "recipient" to stringProp("Telefonnummer oder Kontaktname."),
                    "message" to stringProp("Zu sendender Nachrichtentext.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "play_youtube_music",
                description = "Deutschsprachig: Spiele den gewünschten Musikwunsch möglichst direkt in YouTube Music oder YouTube ab. Nutze dieses Tool nur bei klarer Wiedergabeabsicht wie 'spiel', 'starte Musik' oder 'hör'.",
                parameters = objectSchema(listOf("query"), mapOf("query" to stringProp("Musikwunsch.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "open_rokid_translator",
                description = "Deutschsprachig: Öffne die native Rokid-Echtzeitübersetzer-App oder Szene, falls sie am Telefon installiert ist.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "get_gps_location",
                description = "Deutschsprachig: Lies den letzten bekannten GPS-/Netzwerkstandort des Telefons aus und gib Koordinaten für freihändige Antworten zurück.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "get_weather",
                description = "Deutschsprachig: Hole einen kurzen Wetterbericht für den aktuellen Standort oder einen genannten Ort, ohne einen Browser zu öffnen.",
                parameters = objectSchema(emptyList(), mapOf("location" to stringProp("Optionaler Ort; leer lassen für den aktuellen GPS-Standort.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "web_search",
                description = "Deutschsprachig: Hole eine kurze Websuche- oder Sofortantwort-Zusammenfassung, ohne einen Browser zu öffnen.",
                parameters = objectSchema(listOf("query"), mapOf("query" to stringProp("Suchanfrage.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "set_timer",
                description = "Deutschsprachig: Erstelle freihändig einen Android-Timer.",
                parameters = objectSchema(listOf("seconds"), mapOf(
                    "seconds" to mapOf("type" to "integer", "description" to "Timer-Dauer in Sekunden."),
                    "label" to stringProp("Optionale Timer-Beschriftung.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "create_reminder",
                description = "Deutschsprachig: Speichere eine In-App-Erinnerung, die später aufgelistet werden kann.",
                parameters = objectSchema(listOf("text"), mapOf(
                    "text" to stringProp("Erinnerungstext."),
                    "when" to stringProp("Optionale Fälligkeitszeit in natürlicher Sprache." )
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "list_reminders",
                description = "Deutschsprachig: Lies zuvor gespeicherte Erinnerungen vor.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "create_calendar_event",
                description = "Deutschsprachig: Lege einen Kalendertermin per Sprache an. Nutze ISO-ähnliche Zeiten, wenn möglich.",
                parameters = objectSchema(listOf("title"), mapOf(
                    "title" to stringProp("Titel des Termins."),
                    "start" to stringProp("Optionale Startzeit als Text, z. B. 2026-06-22 14:30."),
                    "end" to stringProp("Optionale Endzeit als Text."),
                    "location" to stringProp("Optionaler Ort."),
                    "notes" to stringProp("Optionale Notizen.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "send_email",
                description = "Deutschsprachig: Öffne eine E-Mail mit Empfänger, Betreff und Text, damit der Nutzer sie sprachgeführt versenden kann.",
                parameters = objectSchema(listOf("to", "subject", "body"), mapOf(
                    "to" to stringProp("E-Mail-Adresse des Empfängers."),
                    "subject" to stringProp("Betreff."),
                    "body" to stringProp("Nachrichtentext.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "start_navigation",
                description = "Deutschsprachig: Geokodiere jedes Navigationsziel zuerst über MapTiler. Starte Navigation nur, wenn MapTiler eine sehr hohe Treffer-Konfidenz liefert; gib sonst die besten Optionen zurück und bitte den Nutzer um Auswahl.",
                parameters = objectSchema(listOf("destination"), mapOf("destination" to stringProp("Zieladresse, Ortsname oder ausgewählte Option, die zuerst über MapTiler geprüft wird.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "open_app",
                description = "Deutschsprachig: Öffne eine installierte App anhand von Paketname oder bekanntem App-Namen.",
                parameters = objectSchema(listOf("app"), mapOf("app" to stringProp("Paketname oder App-Name, z. B. Maps, WhatsApp, Kalender.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "share_text",
                description = "Deutschsprachig: Teile oder diktiere Text über den Android-Teilen-Dialog.",
                parameters = objectSchema(listOf("text"), mapOf("text" to stringProp("Zu teilender Text.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "get_battery_status",
                description = "Deutschsprachig: Lies den Akkustand und Ladezustand des Telefons aus.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "open_accessibility_settings",
                description = "Deutschsprachig: Öffne Bedienungshilfen, falls der Nutzer Freisprech- oder Barrierefreiheitsrechte einrichten will.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "snap_glasses_photo",
                description = "Deutschsprachig: Mache freihändig ein Foto mit der Rokid-Brillenkamera als visuellen Kontext. Die Activity verarbeitet dieses Tool.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "new_chat",
                description = "Deutschsprachig: Erstelle und öffne einen neuen dauerhaften Chat, wenn der Nutzer ein neues Thema, einen neuen Chat oder ein getrenntes Gespräch möchte.",
                parameters = objectSchema(emptyList(), mapOf("title" to stringProp("Optionaler Titel für den neuen Chat.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "list_chats",
                description = "Deutschsprachig: Liste dauerhafte Chats mit IDs, Titeln und Markierung des aktiven Chats auf.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "switch_chat",
                description = "Deutschsprachig: Wechsle vor der Antwort per ID oder exaktem Titel in einen anderen dauerhaften Chat.",
                parameters = objectSchema(emptyList(), mapOf(
                    "chat_id" to stringProp("Chat-ID aus list_chats."),
                    "title" to stringProp("Chat-Titel für den Wechsel, falls keine ID bekannt ist.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "rename_chat",
                description = "Deutschsprachig: Benenne den aktiven oder angegebenen Chat um.",
                parameters = objectSchema(listOf("title"), mapOf(
                    "title" to stringProp("Neuer Titel."),
                    "chat_id" to stringProp("Optionale Chat-ID; leer lassen, um den aktiven Chat umzubenennen.")
                ))
            ))
        )
    }

    private fun executeAssistantTool(toolCall: ToolCall): String {
        val args = parseToolArguments(toolCall.function.arguments)
        listener?.onAssistantToolCall(toolCall.function.name, args)?.let { return it }

        fun arg(name: String): String = args[name].orEmpty()
        fun resolvePhone(recipient: String): String = if (recipient.any { it.isDigit() }) recipient else lookupContactPhone(recipient) ?: recipient

        return when (toolCall.function.name) {
            "place_phone_call" -> placePhoneCall(resolvePhone(arg("recipient")))
            "send_sms" -> sendSms(resolvePhone(arg("recipient")), arg("message"))
            "play_youtube_music" -> playYoutubeMusic(arg("query"))
            "open_rokid_translator" -> openRokidTranslator()
            "get_gps_location" -> getGpsLocation()
            "get_weather" -> getWeather(arg("location"))
            "web_search" -> webSearch(arg("query"))
            "set_timer" -> setTimer(arg("seconds").toIntOrNull() ?: 0, arg("label"))
            "create_reminder" -> createReminder(arg("text"), arg("when"))
            "list_reminders" -> listReminders()
            "create_calendar_event" -> createCalendarEvent(arg("title"), arg("start"), arg("end"), arg("location"), arg("notes"))
            "send_email" -> sendEmail(arg("to"), arg("subject"), arg("body"))
            "start_navigation" -> startNavigation(arg("destination"))
            "open_app" -> openApp(arg("app"))
            "share_text" -> shareText(arg("text"))
            "get_battery_status" -> getBatteryStatus()
            "open_accessibility_settings" -> launchIntent(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS), "Bedienungshilfen")
            "new_chat" -> {
                val session = createChatSession(arg("title"))
                "Neuer Chat '${session.title}' (${session.id}) erstellt und geöffnet."
            }
            "list_chats" -> listChats()
            "switch_chat" -> switchChat(arg("chat_id"), arg("title"))
            "rename_chat" -> renameChat(arg("chat_id"), arg("title"))
            else -> "Unbekanntes Tool: ${toolCall.function.name}"
        }
    }

    private fun parseToolArguments(arguments: String): Map<String, String> {
        val parsed = runCatching { gson.fromJson(arguments, Map::class.java) as Map<*, *> }.getOrDefault(emptyMap<Any, Any>())
        return parsed.mapNotNull { (key, value) -> key?.toString()?.let { it to value?.toString().orEmpty() } }.toMap()
    }

    @Volatile
    private var shouldStopAssistantTurn = false

    private fun launchIntent(intent: Intent, label: String): String = try {
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
        shouldStopAssistantTurn = true
        listener?.onAssistantExternalAppLaunched(label)
        "$label freihändig gestartet."
    } catch (e: Exception) {
        Log.e(appTag, "Could not start $label", e)
        "$label konnte nicht gestartet werden: ${e.message}"
    }

    private fun hasPermission(permission: String): Boolean = context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    private fun lookupContactPhone(name: String): String? = try {
        if (!hasPermission(Manifest.permission.READ_CONTACTS)) return null
        val projection = arrayOf(ContactsContract.CommonDataKinds.Phone.NUMBER)
        val selection = "${ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME_PRIMARY} LIKE ?"
        val selectionArgs = arrayOf("%$name%")
        context.contentResolver.query(
            ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            projection,
            selection,
            selectionArgs,
            null
        )?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        }
    } catch (e: Exception) {
        Log.w(appTag, "Contact lookup failed", e)
        null
    }

    private fun placePhoneCall(phone: String): String {
        if (phone.isBlank()) return "Ich brauche einen Kontaktnamen oder eine Telefonnummer für den Anruf."
        if (!hasPermission(Manifest.permission.CALL_PHONE)) return "Die CALL_PHONE-Berechtigung ist nötig, bevor ich freihändig anrufen kann."
        return launchIntent(Intent(Intent.ACTION_CALL, Uri.parse("tel:${Uri.encode(phone)}")), "Telefonanruf")
    }

    private fun sendSms(phone: String, message: String): String {
        if (phone.isBlank() || message.isBlank()) return "Ich brauche Empfänger und Nachricht, bevor ich eine SMS senden kann."
        if (!hasPermission(Manifest.permission.SEND_SMS)) return "Die SEND_SMS-Berechtigung ist nötig, bevor ich freihändig SMS senden kann."
        return try {
            context.getSystemService(SmsManager::class.java).sendTextMessage(phone, null, message, null, null)
            "SMS an $phone gesendet."
        } catch (e: Exception) {
            Log.e(appTag, "Could not send SMS", e)
            "SMS konnte nicht gesendet werden: ${e.message}"
        }
    }

    private fun playYoutubeMusic(query: String): String {
        if (query.isBlank()) return "Ich brauche ein Lied, einen Künstler, eine Playlist oder ein Genre zum Abspielen."
        val encoded = Uri.encode(query)
        val intents = listOf(
            Intent(Intent.ACTION_VIEW, Uri.parse("https://music.youtube.com/search?q=$encoded")).setPackage("com.google.android.apps.youtube.music"),
            Intent(Intent.ACTION_SEARCH).setPackage("com.google.android.apps.youtube.music").putExtra("query", query),
            Intent(Intent.ACTION_VIEW, Uri.parse("vnd.youtube://results?search_query=$encoded")).setPackage("com.google.android.youtube"),
            Intent(Intent.ACTION_SEARCH).setPackage("com.google.android.youtube").putExtra("query", query),
            Intent(Intent.ACTION_VIEW, Uri.parse("https://www.youtube.com/results?search_query=$encoded&sp=EgIQAQ%253D%253D")),
            Intent(Intent.ACTION_VIEW, Uri.parse("https://music.youtube.com/search?q=$encoded"))
        )
        intents.forEach { intent ->
            if (intent.action == Intent.ACTION_VIEW) intent.addCategory(Intent.CATEGORY_BROWSABLE)
            val result = launchIntent(intent, "YouTube-Wiedergabe für $query")
            if (!result.contains("konnte nicht gestartet werden")) return "$result Falls YouTube nur die Suche öffnet, tippe das erste Ergebnis an; direkte Autoplay-Steuerung ist appabhängig."
        }
        return "YouTube oder YouTube Music konnte nicht gestartet werden."
    }

    private fun openRokidTranslator(): String {
        val packageNames = listOf("com.rokid.translate", "com.rokid.translator", "com.rokid.glass.translator", "com.rokid.ai.translate")
        packageNames.forEach { packageName ->
            context.packageManager.getLaunchIntentForPackage(packageName)?.let { return launchIntent(it, "Rokid-Echtzeitübersetzer") }
        }
        return "Ich konnte die Rokid-Übersetzer-App auf diesem Telefon nicht finden."
    }

    private fun getGpsLocation(): String {
        if (!hasPermission(Manifest.permission.ACCESS_FINE_LOCATION) && !hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)) {
            return "Die Standortberechtigung ist nötig, bevor ich GPS freihändig auslesen kann."
        }
        return try {
            val manager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER)
            val location = providers.asSequence().mapNotNull { provider ->
                runCatching { manager.getLastKnownLocation(provider) }.getOrNull()
            }.maxByOrNull { it.time }
            if (location == null) "Noch kein letzter bekannter Standort verfügbar." else "Aktueller Standort: latitude=${location.latitude}, longitude=${location.longitude}, accuracy=${location.accuracy}m."
        } catch (e: Exception) {
            "Standort konnte nicht gelesen werden: ${e.message}"
        }
    }

    private fun getWeather(location: String): String {
        val query = if (location.isNotBlank()) location else getGpsLocation().takeIf { it.startsWith("Aktueller Standort:") }?.let {
            val lat = Regex("latitude=([-0-9.]+)").find(it)?.groupValues?.getOrNull(1)
            val lon = Regex("longitude=([-0-9.]+)").find(it)?.groupValues?.getOrNull(1)
            if (lat != null && lon != null) "$lat,$lon" else null
        }.orEmpty()
        if (query.isBlank()) return "Ich brauche einen Ort oder die GPS-Berechtigung, um Wetter abzurufen."
        return fetchText("https://wttr.in/${URLEncoder.encode(query, "UTF-8")}?format=3", "Wetter")
    }

    private fun webSearch(query: String): String {
        if (query.isBlank()) return "Ich brauche eine Suchanfrage für die Websuche."
        val url = "https://api.duckduckgo.com/?q=${URLEncoder.encode(query, "UTF-8")}&format=json&no_html=1&skip_disambig=1"
        val raw = fetchText(url, "Websuche")
        val parsed = runCatching { gson.fromJson(raw, Map::class.java) as Map<*, *> }.getOrNull() ?: return raw.take(800)
        val abstract = parsed["AbstractText"]?.toString().orEmpty()
        val heading = parsed["Heading"]?.toString().orEmpty()
        val related = (parsed["RelatedTopics"] as? List<*>)?.firstOrNull()?.let { it as? Map<*, *> }?.get("Text")?.toString().orEmpty()
        return listOf(heading, abstract, related).filter { it.isNotBlank() }.joinToString("\n").ifBlank { "Keine Sofortantwort für $query gefunden." }
    }

    private fun fetchText(url: String, label: String): String = try {
        val request = Request.Builder().url(url).get().build()
        executeTracked(request).use { response ->
            if (!response.isSuccessful) "$label konnte nicht abgerufen werden: HTTP ${response.code}" else response.body?.string()?.trim().orEmpty()
        }
    } catch (e: Exception) {
        "$label konnte nicht abgerufen werden: ${e.message}"
    }

    private fun setTimer(seconds: Int, label: String): String {
        if (seconds <= 0) return "Timer duration must be greater than zero seconds."
        val intent = Intent(AlarmClock.ACTION_SET_TIMER)
            .putExtra(AlarmClock.EXTRA_LENGTH, seconds)
            .putExtra(AlarmClock.EXTRA_MESSAGE, label.ifBlank { "NeuroGlasses timer" })
            .putExtra(AlarmClock.EXTRA_SKIP_UI, true)
        return launchIntent(intent, "${seconds}-second timer")
    }

    private fun createReminder(text: String, dueTime: String): String {
        if (text.isBlank()) return "Reminder text cannot be empty."
        val prefs = context.getSharedPreferences("assistant_reminders", Context.MODE_PRIVATE)
        val existing = prefs.getStringSet("items", emptySet()).orEmpty().toMutableSet()
        existing.add("${System.currentTimeMillis()}|$text|$dueTime")
        prefs.edit().putStringSet("items", existing).apply()
        return "Reminder saved: $text${if (dueTime.isNotBlank()) " ($dueTime)" else ""}."
    }

    private fun listReminders(): String {
        val prefs = context.getSharedPreferences("assistant_reminders", Context.MODE_PRIVATE)
        val reminders = prefs.getStringSet("items", emptySet()).orEmpty().sorted().map { item ->
            val parts = item.split("|", limit = 3)
            "- ${parts.getOrNull(1).orEmpty()}${parts.getOrNull(2)?.takeIf { it.isNotBlank() }?.let { " ($it)" }.orEmpty()}"
        }
        return if (reminders.isEmpty()) "No reminders saved." else reminders.joinToString("\n")
    }

    private fun sendEmail(to: String, subject: String, body: String): String {
        if (to.isBlank()) return "Ich brauche eine E-Mail-Adresse."
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:${Uri.encode(to)}"))
            .putExtra(Intent.EXTRA_SUBJECT, subject)
            .putExtra(Intent.EXTRA_TEXT, body)
        return launchIntent(intent, "E-Mail-Entwurf")
    }

    private data class NavigationCandidate(
        val label: String,
        val latitude: Double,
        val longitude: Double,
        val confidence: Double
    )

    private fun startNavigation(destination: String): String {
        if (destination.isBlank()) return "Ich brauche ein Navigationsziel, das ich zuerst über MapTiler prüfen kann."
        if (getMapTilerApiKey().isBlank()) return "Navigation braucht zuerst einen MapTiler API-Schlüssel in den Einstellungen."

        val candidates = geocodeWithMapTiler(destination)
        if (candidates.isEmpty()) {
            return "MapTiler hat kein passendes Ziel für '$destination' gefunden. Bitte nenne eine genauere Adresse oder einen eindeutigeren Ortsnamen."
        }

        val best = candidates.first()
        val secondConfidence = candidates.getOrNull(1)?.confidence ?: 0.0
        val hasVeryHighConfidence = best.confidence >= 0.90 && (candidates.size == 1 || best.confidence - secondConfidence >= 0.15)
        if (!hasVeryHighConfidence) {
            return candidates.take(3).mapIndexed { index, candidate ->
                "${index + 1}. ${candidate.label} (${(candidate.confidence * 100).toInt()}%)"
            }.joinToString(
                prefix = "MapTiler ist nicht sicher genug. Welche Option soll ich nehmen?\n",
                separator = "\n"
            )
        }

        val encodedLabel = Uri.encode(best.label)
        val coordinateQuery = "${best.latitude},${best.longitude}"
        val intents = listOf(
            Intent(Intent.ACTION_VIEW, Uri.parse("google.navigation:q=$coordinateQuery")).setPackage("com.google.android.apps.maps"),
            Intent(Intent.ACTION_VIEW, Uri.parse("google.navigation:q=$encodedLabel")).setPackage("com.google.android.apps.maps"),
            Intent(Intent.ACTION_VIEW, Uri.parse("geo:${best.latitude},${best.longitude}?q=$encodedLabel"))
        )
        intents.forEach { intent ->
            val result = launchIntent(intent, "Navigation zu ${best.label}")
            if (!result.contains("konnte nicht gestartet werden")) return "$result MapTiler-Treffer: ${best.label} (${(best.confidence * 100).toInt()}%)."
        }
        return "MapTiler hat '${best.label}' sicher gefunden, aber die Navigation konnte nicht gestartet werden."
    }

    private fun geocodeWithMapTiler(query: String): List<NavigationCandidate> {
        val apiKey = getMapTilerApiKey()
        if (apiKey.isBlank()) {
            Log.w(appTag, "MapTiler API key missing; navigation cannot be geocoded")
            return emptyList()
        }

        val url = "https://api.maptiler.com/geocoding/${URLEncoder.encode(query, "UTF-8")}.json?key=${URLEncoder.encode(apiKey, "UTF-8")}&limit=3&language=de"
        return try {
            val request = Request.Builder().url(url).get().build()
            executeTracked(request).use { response ->
                if (!response.isSuccessful) {
                    Log.w(appTag, "MapTiler geocoding failed: HTTP ${response.code}")
                    return emptyList()
                }
                val body = response.body?.string().orEmpty()
                val parsed = gson.fromJson(body, Map::class.java) as? Map<*, *> ?: return emptyList()
                val features = parsed["features"] as? List<*> ?: return emptyList()
                features.mapNotNull { feature ->
                    val item = feature as? Map<*, *> ?: return@mapNotNull null
                    val center = item["center"] as? List<*> ?: return@mapNotNull null
                    val longitude = (center.getOrNull(0) as? Number)?.toDouble() ?: return@mapNotNull null
                    val latitude = (center.getOrNull(1) as? Number)?.toDouble() ?: return@mapNotNull null
                    val label = item["place_name"]?.toString()?.takeIf { it.isNotBlank() }
                        ?: item["text"]?.toString()?.takeIf { it.isNotBlank() }
                        ?: return@mapNotNull null
                    val confidence = ((item["relevance"] as? Number)?.toDouble()
                        ?: (item["score"] as? Number)?.toDouble()
                        ?: 0.0).coerceIn(0.0, 1.0)
                    NavigationCandidate(label, latitude, longitude, confidence)
                }.sortedByDescending { it.confidence }
            }
        } catch (e: Exception) {
            Log.e(appTag, "MapTiler geocoding error", e)
            emptyList()
        }
    }

    private fun openApp(app: String): String {
        if (app.isBlank()) return "Ich brauche einen App-Namen."
        val normalized = app.lowercase(Locale.ROOT)
        val aliases = mapOf(
            "maps" to "com.google.android.apps.maps",
            "google maps" to "com.google.android.apps.maps",
            "whatsapp" to "com.whatsapp",
            "kalender" to "com.google.android.calendar",
            "calendar" to "com.google.android.calendar",
            "youtube" to "com.google.android.youtube",
            "gmail" to "com.google.android.gm"
        )
        val packageName = aliases[normalized] ?: app
        val intent = context.packageManager.getLaunchIntentForPackage(packageName)
            ?: return "Ich konnte $app auf diesem Telefon nicht finden."
        return launchIntent(intent, app)
    }

    private fun shareText(text: String): String {
        if (text.isBlank()) return "Ich brauche Text zum Teilen."
        val sendIntent = Intent(Intent.ACTION_SEND)
            .setType("text/plain")
            .putExtra(Intent.EXTRA_TEXT, text)
        return launchIntent(Intent.createChooser(sendIntent, "Text teilen"), "Teilen")
    }

    private fun getBatteryStatus(): String {
        val batteryIntent = context.registerReceiver(null, android.content.IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = batteryIntent?.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = batteryIntent?.getIntExtra(android.os.BatteryManager.EXTRA_SCALE, -1) ?: -1
        val status = batteryIntent?.getIntExtra(android.os.BatteryManager.EXTRA_STATUS, -1) ?: -1
        val percent = if (level >= 0 && scale > 0) (level * 100 / scale) else -1
        val charging = status == android.os.BatteryManager.BATTERY_STATUS_CHARGING || status == android.os.BatteryManager.BATTERY_STATUS_FULL
        return if (percent >= 0) "Akkustand: $percent%, lädt: ${if (charging) "ja" else "nein"}." else "Akkustand ist nicht verfügbar."
    }

    private fun createCalendarEvent(title: String, start: String, end: String, location: String, notes: String): String {
        if (title.isBlank()) return "Ich brauche einen Termintitel."
        val intent = Intent(Intent.ACTION_INSERT)
            .setData(CalendarContract.Events.CONTENT_URI)
            .putExtra(CalendarContract.Events.TITLE, title)
            .putExtra(CalendarContract.Events.EVENT_LOCATION, location)
            .putExtra(CalendarContract.Events.DESCRIPTION, listOf(notes, start.takeIf { it.isNotBlank() }?.let { "Start: $it" }, end.takeIf { it.isNotBlank() }?.let { "Ende: $it" }).filterNotNull().joinToString("\n"))
        return launchIntent(intent, "Kalendertermin")
    }

    /**
     * Call OpenAI API for chat completion with streaming
     * @param instruction The user instruction/prompt
     * @param image Optional image to include in the request (for vision models)
     */
    fun callOpenAIStreaming(instruction: String, image: Bitmap?) {
        Log.d(appTag, "OpenAI Streaming API called")
        Log.d(appTag, "Instruction: $instruction")
        Log.d(appTag, "Has image: ${image != null}")

        Thread {
            try {
                // Build the message content
                val contentList = mutableListOf<Content>()

                // Add text instruction
                contentList.add(Content(type = "text", text = instruction))

                // Add image if provided
                if (image != null) {
                    val base64Image = bitmapToBase64(image)
                    val imageDataUrl = "data:image/png;base64,$base64Image"
                    contentList.add(
                        Content(
                            type = "image_url",
                            imageUrl = ImageUrl(url = imageDataUrl)
                        )
                    )
                }

                val requestChatId = activeChatId()

                // Build the messages list with system prompt, persisted chat history, and current user turn.
                val messagesList = mutableListOf<Message>()

                // Add system message
                val systemPrompt = getSystemPrompt() + "\n\nAntworte immer auf Deutsch (Österreich), knapp und freihändig nutzbar. Erkenne die Absicht statt nur Schlüsselwörter zu lesen: Bei visuellen Fragen wie Finger zählen, Text lesen, Objekte/Farben erkennen oder \"was sehe ich\" soll ein Foto als Kontext genutzt werden; wenn noch kein Bild vorhanden ist, nutze snap_glasses_photo. Bei Verabschiedungen wie \"tschüss\", \"auf Wiedersehen\" oder \"goodbye\" verabschiede dich kurz und beende das Gespräch. Nutze verfügbare Tools proaktiv, aber nur wenn die Absicht eindeutig und die Aktion nicht destruktiv ist. Frage nur nach, wenn für Anruf, SMS, Kalender oder E-Mail Pflichtangaben fehlen oder die Aktion unsicher wäre. Navigation ist als start_navigation Tool verfügbar. Jede Navigationsabsicht muss start_navigation nutzen, damit MapTiler zuerst geokodiert. Starte erst, wenn das Tool eine sehr hohe MapTiler-Konfidenz meldet; wenn das Tool Optionen zurückgibt, zeige diese Optionen und warte auf die Auswahl des Nutzers. Du hast persistente, mehrere Chat-Verläufe. Nutze new_chat, list_chats, switch_chat und rename_chat, wenn der Nutzer neue Chats, getrennte Themen, Chatlisten, Umbenennungen oder Chatwechsel wünscht."
                messagesList.add(
                    Message(
                        role = "system",
                        content = listOf(Content(type = "text", text = systemPrompt))
                    )
                )

                // Add persisted chat history before the current turn. Image content is not persisted.
                messagesList.addAll(activeChatHistoryMessages())

                // Add user message
                messagesList.add(
                    Message(
                        role = "user",
                        content = contentList
                    )
                )

                shouldStopAssistantTurn = false
                val toolMessages = if (image != null) {
                    // Vision requests must carry the actual image data in the final chat turn.
                    // Do not run a separate tool-planning chat call first, because some
                    // providers/models drop or cannot inspect image content there and then the
                    // streamed answer can incorrectly claim that no image data was received.
                    messagesList
                } else {
                    resolveToolCallsIfNeeded(messagesList)
                }
                if (shouldStopAssistantTurn) {
                    Log.i(appTag, "Assistant turn stopped after launching external app")
                    return@Thread
                }

                // Create the request with streaming enabled. Tools are resolved in a prior
                // non-streaming pass so the final response can stream cleanly to the glasses.
                val request = OpenAIRequest(
                    model = getVlmModel(),
                    messages = toolMessages,
                    maxTokens = getVlmMaxTokens(),
                    stream = true
                )

                // Convert request to JSON
                val jsonBody = gson.toJson(request)
                Log.d(appTag, "Request JSON: $jsonBody")

                // Create HTTP request
                val requestBody = jsonBody.toRequestBody("application/json".toMediaType())
                val httpRequest = Request.Builder()
                    .url(getChatApiUrl())
                    .addHeader("Content-Type", "application/json")
                    .addHeader("Authorization", "Bearer ${getApiToken()}")
                    .post(requestBody)
                    .build()

                // Notify streaming started
                listener?.onOpenAIStreamingStarted()

                // Execute the request with streaming-specific timeout
                executeTracked(httpRequest, isStreaming = true).use { response ->
                    if (!response.isSuccessful) {
                        val errorBody = response.body?.string() ?: "Unknown error"
                        Log.e(appTag, "API call failed: ${response.code} - $errorBody")
                        listener?.onOpenAIFailed("API call failed: ${response.code}")
                        return@Thread
                    }

                    // Read the streaming response
                    val source = response.body?.source()
                    if (source == null) {
                        Log.e(appTag, "Empty response body")
                        listener?.onOpenAIFailed("Empty response")
                        return@Thread
                    }

                    val fullResponse = StringBuilder()
                    okio.Buffer()

                    // Parse Server-Sent Events with buffered reading to avoid blocking on newlines
                    try {
                        while (!source.exhausted()) {
                            // Read available data without blocking indefinitely
                            // This uses a buffered approach that won't wait for newlines
                            val line = try {
                                source.readUtf8LineStrict()
                            } catch (_: java.io.EOFException) {
                                // Handle case where stream ends without final newline
                                Log.d(appTag, "Stream ended")
                                break
                            }

                            if (line.isEmpty()) {
                                // SSE uses blank lines as delimiters between messages
                                continue
                            }

                            Log.v(appTag, "Raw SSE line: $line")

                            // SSE format: "data: {json}"
                            if (line.startsWith("data: ")) {
                                val data = line.substring(6).trim()

                                // Check for [DONE] marker
                                if (data == "[DONE]") {
                                    Log.d(appTag, "Streaming completed with [DONE] marker")
                                    listener?.onOpenAIStreamingChunk("", true)
                                    break
                                }

                                // Skip empty data lines
                                if (data.isEmpty()) {
                                    continue
                                }

                                try {
                                    // Parse the JSON chunk
                                    val streamingResponse = gson.fromJson(data, StreamingResponse::class.java)
                                    val delta = streamingResponse.choices.firstOrNull()?.delta
                                    val content = delta?.content

                                    if (content != null && content.isNotEmpty()) {
                                        fullResponse.append(content)
                                        Log.d(appTag, "Received chunk: $content")

                                        // Check if this is the final chunk
                                        val finishReason = streamingResponse.choices.firstOrNull()?.finishReason
                                        val isComplete = finishReason != null

                                        listener?.onOpenAIStreamingChunk(content, isComplete)
                                    }
                                } catch (e: Exception) {
                                    Log.e(appTag, "Error parsing streaming chunk: ${e.message}, data: $data")
                                    // Continue processing remaining chunks
                                }
                            }
                        }
                    } catch (_: java.net.SocketTimeoutException) {
                        Log.e(appTag, "Streaming read timeout - this may indicate the server is not sending data in SSE format")
                        listener?.onOpenAIFailed("Streaming timeout: server may not be sending proper SSE format")
                        return@Thread
                    }

                    // Also send the full response to the non-streaming callback for backwards compatibility
                    val finalResponse = fullResponse.toString()
                    if (finalResponse.isNotEmpty()) {
                        Log.d(appTag, "Full streaming response: $finalResponse")
                        appendChatMessages(
                            activeChatId().takeIf { it != requestChatId } ?: requestChatId,
                            listOf(
                                PersistedChatMessage("user", instruction),
                                PersistedChatMessage("assistant", finalResponse)
                            )
                        )
                        listener?.onOpenAIResponse(finalResponse)
                    } else {
                        Log.w(appTag, "Streaming completed but no content was received")
                    }
                }
            } catch (e: IOException) {
                Log.e(appTag, "Network error: ${e.message}", e)
                listener?.onOpenAIFailed("Network error: ${e.message}")
            } catch (e: Exception) {
                Log.e(appTag, "Error calling OpenAI Streaming API: ${e.message}", e)
                listener?.onOpenAIFailed("Error: ${e.message}")
            }
        }.start()
    }


    private fun resolveToolCallsIfNeeded(messagesList: MutableList<Message>): List<Message> {
        val request = OpenAIRequest(
            model = getVlmModel(),
            messages = messagesList,
            maxTokens = getVlmMaxTokens(),
            stream = false,
            tools = assistantTools(),
            toolChoice = "auto"
        )
        val jsonBody = gson.toJson(request)
        val requestBody = jsonBody.toRequestBody("application/json".toMediaType())
        val httpRequest = Request.Builder()
            .url(getChatApiUrl())
            .addHeader("Content-Type", "application/json")
            .addHeader("Authorization", "Bearer ${getApiToken()}")
            .post(requestBody)
            .build()

        return try {
            executeTracked(httpRequest).use { response ->
                if (!response.isSuccessful) {
                    Log.w(appTag, "Tool planning skipped: ${response.code} - ${response.body?.string()}")
                    return messagesList
                }
                val responseBody = response.body?.string() ?: return messagesList
                val chatResponse = gson.fromJson(responseBody, ChatCompletionResponse::class.java)
                val assistantMessage = chatResponse.choices.firstOrNull()?.message ?: return messagesList
                val toolCalls = assistantMessage.toolCalls.orEmpty()
                if (toolCalls.isEmpty()) {
                    messagesList.add(assistantMessage)
                    return messagesList
                }

                messagesList.add(assistantMessage)
                toolCalls.forEach { toolCall ->
                    val result = executeAssistantTool(toolCall)
                    messagesList.add(
                        Message(
                            role = "tool",
                            content = result,
                            toolCallId = toolCall.id
                        )
                    )
                }
                messagesList
            }
        } catch (e: Exception) {
            Log.e(appTag, "Tool planning failed; continuing without tools", e)
            messagesList
        }
    }

    /**
     * Convert Bitmap to base64 string
     * @param bitmap The bitmap to convert
     * @return Base64 encoded string
     */
    private fun bitmapToBase64(bitmap: Bitmap): String {
        val byteArrayOutputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)
        val byteArray = byteArrayOutputStream.toByteArray()
        return Base64.encodeToString(byteArray, Base64.NO_WRAP)
    }

    /**
     * Call TTS API to convert text to speech with streaming support
     * @param text The text to convert to speech
     * @param outputDir The directory to save the audio file
     * @param streaming If true, streams audio chunks as they arrive; if false, waits for complete file
     */
    fun callTtsAPI(text: String, outputDir: File, streaming: Boolean = true) {
        Log.d(appTag, "TTS API called with text: $text (streaming: $streaming)")

        if (!outputDir.exists()) {
            outputDir.mkdirs()
        }

        Thread {
            try {
                // Create the TTS request
                val request = TtsRequest(
                    model = getTtsModel(),
                    input = text,
                    voice = getTtsVoice()
                )

                // Convert request to JSON
                val jsonBody = gson.toJson(request)
                Log.d(appTag, "TTS Request JSON: $jsonBody")

                // Create HTTP request
                val requestBody = jsonBody.toRequestBody("application/json".toMediaType())
                val httpRequest = Request.Builder()
                    .url(getTtsApiUrl())
                    .addHeader("Content-Type", "application/json")
                    .addHeader("Authorization", "Bearer ${getApiToken()}")
                    .post(requestBody)
                    .build()

                if (streaming) {
                    // Streaming mode: send chunks as they arrive
                    callTtsAPIStreaming(httpRequest, outputDir)
                } else {
                    // Non-streaming mode: wait for complete file
                    callTtsAPINonStreaming(httpRequest, outputDir)
                }
            } catch (e: IOException) {
                Log.e(appTag, "Network error during TTS: ${e.message}", e)
                listener?.onTtsFailed("Network error: ${e.message}")
            } catch (e: Exception) {
                Log.e(appTag, "Error calling TTS API: ${e.message}", e)
                listener?.onTtsFailed("Error: ${e.message}")
            }
        }.start()
    }

    /**
     * Call TTS API in streaming mode
     */
    private fun callTtsAPIStreaming(httpRequest: Request, outputDir: File) {
        // Execute the request with streaming
        executeTracked(httpRequest, isStreaming = true).use { response ->
            val contentType = response.body?.contentType()
            val contentLength = response.body?.contentLength()
            Log.d(appTag, "TTS response received: code=${response.code}, contentType=$contentType, contentLength=$contentLength")

            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                Log.e(appTag, "TTS API call failed: ${response.code} - $errorBody")
                listener?.onTtsFailed("TTS API call failed: ${response.code}")
                return
            }

            // Check if response is actually audio
            if (contentType?.toString()?.startsWith("text/") == true) {
                // Server returned text instead of audio - likely an error message
                val errorText = response.body?.string() ?: "Unknown error"
                Log.e(appTag, "TTS API returned text instead of audio: $errorText")
                listener?.onTtsFailed("TTS API error: $errorText")
                return
            }

            // Notify streaming started
            listener?.onTtsStreamingStarted()
            Log.d(appTag, "TTS streaming started")

            // Read the streaming response
            val inputStream = response.body?.byteStream()
            if (inputStream == null) {
                Log.e(appTag, "Empty TTS response body")
                listener?.onTtsFailed("Empty audio response")
                return
            }

            Log.d(appTag, "Input stream obtained, starting to read chunks...")

            // Create file to save complete audio for later use
            val timestamp = System.currentTimeMillis()
            val audioFile = File(outputDir, "tts_result_$timestamp.mp3")
            val fileOutputStream = audioFile.outputStream()

            try {
                // Stream audio data in chunks
                val buffer = ByteArray(4096) // 4KB chunks
                var bytesRead: Int
                var totalBytes = 0L
                var chunkCount = 0

                Log.d(appTag, "Starting read loop...")
                while (inputStream.read(buffer).also { bytesRead = it } != -1) {
                    chunkCount++
                    Log.d(appTag, "Read attempt #$chunkCount: $bytesRead bytes")

                    if (bytesRead > 0) {
                        // Create a chunk of the exact size read
                        val chunk = buffer.copyOf(bytesRead)

                        // Save to file
                        fileOutputStream.write(chunk)
                        totalBytes += bytesRead

                        // Send chunk to listener for immediate playback
                        listener?.onTtsStreamingChunk(chunk, false)

                        Log.d(appTag, "TTS chunk received: $bytesRead bytes (total: $totalBytes)")
                    }
                }

                Log.d(appTag, "Read loop completed after $chunkCount iterations")

                // Notify streaming complete
                listener?.onTtsStreamingChunk(ByteArray(0), true)
                Log.d(appTag, "TTS streaming completed: $totalBytes bytes")

                // Close file stream
                fileOutputStream.close()

                // Notify that the complete file is ready
                Log.d(appTag, "TTS audio saved to: ${audioFile.absolutePath} ($totalBytes bytes)")
                listener?.onTtsComplete(audioFile)

            } catch (e: Exception) {
                fileOutputStream.close()
                Log.e(appTag, "Error during TTS streaming: ${e.message}", e)
                listener?.onTtsFailed("Streaming error: ${e.message}")
            }
        }
    }

    /**
     * Call TTS API in non-streaming mode (original behavior)
     */
    private fun callTtsAPINonStreaming(httpRequest: Request, outputDir: File) {
        // Execute the request
        executeTracked(httpRequest).use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                Log.e(appTag, "TTS API call failed: ${response.code} - $errorBody")
                listener?.onTtsFailed("TTS API call failed: ${response.code}")
                return
            }

            // Save the audio data to file
            val audioBytes = response.body?.bytes()
            if (audioBytes != null && audioBytes.isNotEmpty()) {
                val timestamp = System.currentTimeMillis()
                val audioFile = File(outputDir, "tts_result_$timestamp.mp3")

                audioFile.outputStream().use { fileOut ->
                    fileOut.write(audioBytes)
                }

                Log.d(appTag, "TTS audio saved to: ${audioFile.absolutePath} (${audioBytes.size} bytes)")
                listener?.onTtsComplete(audioFile)
            } else {
                Log.e(appTag, "Empty TTS response body")
                listener?.onTtsFailed("Empty audio response")
            }
        }
    }

    /**
     * Release resources
     */
    fun release() {
        listener = null
        Log.d(appTag, "OpenAIHelper released")
    }
}
