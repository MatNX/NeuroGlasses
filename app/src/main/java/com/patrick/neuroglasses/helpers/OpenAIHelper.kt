package com.patrick.neuroglasses.helpers

import android.annotation.SuppressLint
import android.Manifest
import android.content.Context
import android.content.Intent
import android.app.KeyguardManager
import android.net.Uri
import android.os.Bundle
import android.provider.AlarmClock
import android.provider.CalendarContract
import android.provider.Settings
import android.provider.ContactsContract
import android.telephony.PhoneNumberUtils
import android.telephony.SmsManager
import android.telecom.TelecomManager
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
    val voice: String,
    @SerializedName("response_format")
    val responseFormat: String = "wav"
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

    private companion object {
        private const val MAX_TOOL_ROUNDS = 3
        private const val MAX_TTS_INPUT_CHARS = 200
        private const val MIN_TTS_AUDIO_BYTES = 44
    }

    private data class ContactPhone(
        val displayName: String,
        val number: String
    )

    private data class PhoneTarget(
        val original: String,
        val number: String?,
        val displayName: String? = null,
        val contactPermissionMissing: Boolean = false
    )

    private data class ToolResolutionResult(
        val messages: List<Message>,
        val results: List<String>
    )

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
        } ?: return "Chat not found. Use list_chats to see available chat ids and titles."
        chatPrefs.edit().putString("active_chat_id", session.id).apply()
        return "Switched to chat '${session.title}' (${session.id})."
    }

    private fun listChats(): String {
        val active = activeChatId()
        val sessions = getChatSessions()
        if (sessions.isEmpty()) return "No chats exist yet."
        return sessions.joinToString("\n") { session ->
            val marker = if (session.id == active) "*" else "-"
            "$marker ${session.title} [${session.id}] (${session.messages.size} messages)"
        }
    }

    private fun renameChat(chatId: String?, title: String): String {
        if (title.isBlank()) return "A new chat title is required."
        val targetId = chatId?.takeIf { it.isNotBlank() } ?: activeChatId()
        val updated = updateChatSession(targetId) { it.copy(title = title) }
        return if (updated == null) "Chat not found." else "Renamed chat to '${updated.title}'."
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
         * Called after a tool call finishes so the UI can show progress before the final answer.
         */
        fun onAssistantToolResult(toolName: String, result: String) {}

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
                getClient().newCall(request).execute().use { response ->
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


    private fun assistantTools(
        instruction: String,
        allowGlassesPhotoTool: Boolean,
        includeChatTools: Boolean
    ): List<AssistantTool> {
        fun objectSchema(required: List<String>, properties: Map<String, Map<String, Any>>): Map<String, Any> = mapOf(
            "type" to "object",
            "required" to required,
            "properties" to properties,
            "additionalProperties" to false
        )

        fun stringProp(description: String): Map<String, Any> = mapOf("type" to "string", "description" to description)

        val tools = listOf(
            AssistantTool(function = ToolFunction(
                name = "place_phone_call",
                description = "Hands-free: place a phone call. A contact name is enough; do not ask for a phone number when the user supplied a name. The app resolves contact names and falls back to the dialer if CALL_PHONE permission is missing.",
                parameters = objectSchema(listOf("recipient"), mapOf("recipient" to stringProp("Phone number or contact name; names are valid recipients.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "send_sms",
                description = "Hands-free: send an SMS. A contact name is enough; do not ask for a phone number when the user supplied a name. The app resolves contact names and falls back to the SMS composer if direct sending is unavailable.",
                parameters = objectSchema(listOf("recipient", "message"), mapOf(
                    "recipient" to stringProp("Phone number or contact name; names are valid recipients."),
                    "message" to stringProp("Message body to send.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "find_contact_phone",
                description = "Look up a contact name in the phone contacts and return the best matching phone number before calling or texting.",
                parameters = objectSchema(listOf("recipient"), mapOf("recipient" to stringProp("Contact name to look up.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "stop_conversation",
                description = "Close the active NeuroGlasses conversation when the user explicitly says stop, cancel, goodbye, or end the session.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "start_navigation",
                description = "Start turn-by-turn navigation to a destination, preferring a maps app that can begin navigation directly.",
                parameters = objectSchema(listOf("destination"), mapOf("destination" to stringProp("Address, landmark, or business name.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "play_youtube_music",
                description = "Start YouTube/YouTube Music playback or search for a requested song, artist, playlist, or genre.",
                parameters = objectSchema(listOf("query"), mapOf("query" to stringProp("Music query to play.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "open_rokid_translator",
                description = "Open the native Rokid real-time translator app/scene if it is installed on the companion phone.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "get_gps_location",
                description = "Read the phone's last known GPS/network location and return coordinates for the assistant to use hands-free.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "get_weather",
                description = "Fetch a concise weather report for the current GPS location or a named place without opening a browser.",
                parameters = objectSchema(emptyList(), mapOf("location" to stringProp("Optional city/place; omit for current GPS location.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "web_search",
                description = "Fetch a short web-search/instant-answer summary without opening a browser.",
                parameters = objectSchema(listOf("query"), mapOf("query" to stringProp("Search query.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "set_timer",
                description = "Create a hands-free Android timer.",
                parameters = objectSchema(listOf("seconds"), mapOf(
                    "seconds" to mapOf("type" to "integer", "description" to "Timer duration in seconds."),
                    "label" to stringProp("Optional timer label.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "create_reminder",
                description = "Store an in-app reminder that can be listed later by the assistant.",
                parameters = objectSchema(listOf("text"), mapOf(
                    "text" to stringProp("Reminder text."),
                    "when" to stringProp("Optional natural language due time." )
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "list_reminders",
                description = "Read reminders previously stored by the assistant.",
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
                description = "Create and switch to a fresh persistent chat thread when the user asks for a new topic, new chat, or separate conversation.",
                parameters = objectSchema(emptyList(), mapOf("title" to stringProp("Optional title for the new chat.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "list_chats",
                description = "List persistent chat threads with ids, titles, and the active chat marker.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "switch_chat",
                description = "Switch the active persistent chat thread by id or exact title before answering.",
                parameters = objectSchema(emptyList(), mapOf(
                    "chat_id" to stringProp("Chat id returned by list_chats."),
                    "title" to stringProp("Chat title to switch to if no id is known.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "rename_chat",
                description = "Rename the active persistent chat or a specified chat.",
                parameters = objectSchema(listOf("title"), mapOf(
                    "title" to stringProp("New title."),
                    "chat_id" to stringProp("Optional chat id; omit to rename active chat.")
                ))
            ))
        )

        return tools.filter { tool ->
            when (tool.function.name) {
                "new_chat", "list_chats", "switch_chat", "rename_chat" -> includeChatTools
                "snap_glasses_photo" -> allowGlassesPhotoTool
                "open_rokid_translator" -> hasAnyKeyword(
                    instruction,
                    "open translator", "open the translator", "start translator", "launch translator",
                    "öffne übersetzer", "öffne den übersetzer", "starte übersetzer",
                    "starte den übersetzer", "öffne rokid translator", "starte rokid translator"
                )
                "open_app" -> hasAnyKeyword(instruction, "öffne", "open", "starte", "start", "launch", "app")
                else -> true
            }
        }
    }

    private fun shouldResolveTools(instruction: String): Boolean {
        val normalized = instruction.trim().lowercase(Locale.ROOT)
        if (normalized.isBlank()) return false
        if (isProbeInstruction(instruction)) return false

        return hasAnyKeyword(
            normalized,
            "ruf", "anruf", "call", "sms", "nachricht", "message", "navig", "route", "maps",
            "musik", "youtube", "wetter", "weather", "suche", "search", "web", "timer",
            "erinner", "remind", "kalender", "termin", "calendar", "mail", "email", "e-mail",
            "öffne", "open", "starte", "app", "teile", "share", "akku", "battery",
            "foto", "photo", "bild", "camera", "kamera", "übersetz", "translate", "translator",
            "chat", "liste chats", "neuer chat", "stop", "stopp", "beende", "beenden",
            "abbrechen", "cancel", "tschüss", "tschuess", "goodbye"
        )
    }

    private fun isProbeInstruction(instruction: String): Boolean =
        instruction.trim().lowercase(Locale.ROOT) in setOf(
            "test", "testing", "probe", "check", "ping", "hi", "hallo", "hello"
        )

    private fun hasAnyKeyword(text: String, vararg keywords: String): Boolean {
        val normalized = text.lowercase(Locale.ROOT)
        return keywords.any { normalized.contains(it.lowercase(Locale.ROOT)) }
    }

    private fun executeAssistantTool(toolCall: ToolCall): String {
        val args = parseToolArguments(toolCall.function.arguments)
        listener?.onAssistantToolCall(toolCall.function.name, args)?.let { return it }

        fun arg(name: String): String = args[name].orEmpty()

        return when (toolCall.function.name) {
            "place_phone_call" -> placePhoneCall(arg("recipient"))
            "send_sms" -> sendSms(arg("recipient"), arg("message"))
            "find_contact_phone" -> findContactPhone(arg("recipient"))
            "stop_conversation" -> "Conversation stopped."
            "start_navigation" -> launchIntent(Intent(Intent.ACTION_VIEW, Uri.parse("google.navigation:q=${Uri.encode(arg("destination"))}")), "turn-by-turn navigation")
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
            "open_app" -> openApp(arg("app"))
            "share_text" -> shareText(arg("text"))
            "get_battery_status" -> getBatteryStatus()
            "open_accessibility_settings" -> launchIntent(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS), "Bedienungshilfen")
            "new_chat" -> {
                val session = createChatSession(arg("title"))
                "Created and switched to new chat '${session.title}' (${session.id})."
            }
            "list_chats" -> listChats()
            "switch_chat" -> switchChat(arg("chat_id"), arg("title"))
            "rename_chat" -> renameChat(arg("chat_id"), arg("title"))
            else -> "Unknown tool: ${toolCall.function.name}"
        }
    }

    private fun parseToolArguments(arguments: String): Map<String, String> {
        val parsed = runCatching { gson.fromJson(arguments, Map::class.java) as Map<*, *> }.getOrDefault(emptyMap<Any, Any>())
        return parsed.mapNotNull { (key, value) -> key?.toString()?.let { it to value?.toString().orEmpty() } }.toMap()
    }

    private fun buildToolResultFallback(results: List<String>): String {
        if (results.isEmpty()) return ""
        val lastResult = results.last().substringAfter(":").trim()
        return when {
            lastResult.startsWith("Started ", ignoreCase = true) -> "Erledigt: ${lastResult.removePrefix("Started ").removeSuffix(" hands-free.")} gestartet."
            lastResult.startsWith("Could not", ignoreCase = true) -> "Das hat leider nicht geklappt: $lastResult"
            lastResult.contains("permission", ignoreCase = true) -> "Dafür fehlt noch eine Berechtigung: $lastResult"
            else -> lastResult
        }
    }

    private fun launchIntent(intent: Intent, label: String): String = try {
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        context.startActivity(intent)
        "Started $label hands-free."
    } catch (e: Exception) {
        Log.e(appTag, "Could not start $label", e)
        val lockedSuffix = if (isDeviceLocked()) " The phone is locked; unlock it and try again if Android blocked the target app." else ""
        "Could not start $label: ${e.message}.$lockedSuffix"
    }

    private fun hasPermission(permission: String): Boolean = context.checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED

    private fun isDeviceLocked(): Boolean =
        context.getSystemService(KeyguardManager::class.java)?.isKeyguardLocked == true

    private fun lookupContactPhone(name: String): ContactPhone? {
        val query = normalizeContactText(name)
        if (query.isBlank()) return null
        if (!hasPermission(Manifest.permission.READ_CONTACTS)) return null

        return try {
            val projection = arrayOf(
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME_PRIMARY,
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
                ContactsContract.CommonDataKinds.Phone.NUMBER
            )
            val matches = mutableListOf<Pair<Int, ContactPhone>>()
            context.contentResolver.query(
                ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                projection,
                null,
                null,
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME_PRIMARY
            )?.use { cursor ->
                val primaryNameIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME_PRIMARY)
                val nameIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
                val numberIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
                val displayNameIndex = if (primaryNameIndex >= 0) primaryNameIndex else nameIndex
                if (displayNameIndex < 0 || numberIndex < 0) return@use

                while (cursor.moveToNext()) {
                    val displayName = cursor.getString(displayNameIndex).orEmpty()
                    val number = cursor.getString(numberIndex).orEmpty()
                    if (displayName.isBlank() || number.isBlank()) continue

                    val normalizedName = normalizeContactText(displayName)
                    val score = contactMatchScore(query, normalizedName) ?: continue
                    val normalizedNumber = PhoneNumberUtils.normalizeNumber(number).ifBlank { number }
                    matches += score to ContactPhone(displayName, normalizedNumber)
                }
            }
            matches.sortedWith(compareBy<Pair<Int, ContactPhone>> { it.first }.thenBy { it.second.displayName.length })
                .firstOrNull()
                ?.second
        } catch (e: Exception) {
            Log.w(appTag, "Contact lookup failed", e)
            null
        }
    }

    private fun normalizeContactText(value: String): String =
        value.lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N}\\s]+"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun contactMatchScore(query: String, candidate: String): Int? {
        if (candidate == query) return 0
        if (candidate.startsWith(query)) return 1
        if (candidate.contains(query)) return 2
        val terms = query.split(" ").filter { it.isNotBlank() }
        if (terms.isNotEmpty() && terms.all { candidate.contains(it) }) return 3
        return null
    }

    private fun resolvePhoneTarget(recipient: String): PhoneTarget {
        val trimmed = recipient.trim()
        if (trimmed.isBlank()) return PhoneTarget(original = recipient, number = null)
        if (trimmed.any { it.isDigit() }) {
            return PhoneTarget(original = recipient, number = PhoneNumberUtils.normalizeNumber(trimmed).ifBlank { trimmed })
        }

        val contact = lookupContactPhone(trimmed)
        if (contact != null) {
            return PhoneTarget(original = recipient, number = contact.number, displayName = contact.displayName)
        }

        return PhoneTarget(
            original = recipient,
            number = null,
            contactPermissionMissing = !hasPermission(Manifest.permission.READ_CONTACTS)
        )
    }

    private fun findContactPhone(recipient: String): String {
        val target = resolvePhoneTarget(recipient)
        return when {
            target.number != null && target.displayName != null -> "Found ${target.displayName}: ${target.number}."
            target.number != null -> "Recipient is already a phone number: ${target.number}."
            target.contactPermissionMissing -> "READ_CONTACTS permission is required before I can look up $recipient."
            else -> "I could not find a phone number for $recipient."
        }
    }

    @SuppressLint("MissingPermission")
    private fun placePhoneCall(recipient: String): String {
        val target = resolvePhoneTarget(recipient)
        val phone = target.number
        if (phone.isNullOrBlank()) {
            return if (target.contactPermissionMissing) {
                "READ_CONTACTS permission is required before I can find $recipient. Provide a phone number or grant contacts permission."
            } else {
                "I could not find a phone number for $recipient."
            }
        }

        val label = target.displayName?.let { "phone call to $it" } ?: "phone call"
        if (hasPermission(Manifest.permission.CALL_PHONE)) {
            val callUri = Uri.parse("tel:${Uri.encode(phone)}")
            val telecomManager = context.getSystemService(TelecomManager::class.java)
            if (telecomManager != null) {
                runCatching {
                    telecomManager.placeCall(callUri, Bundle.EMPTY)
                }.onSuccess {
                    return "Started $label hands-free."
                }.onFailure { error ->
                    Log.e(appTag, "TelecomManager.placeCall failed", error)
                }
            }

            return launchIntent(Intent(Intent.ACTION_CALL, callUri), label)
        }

        val dialResult = launchIntent(Intent(Intent.ACTION_DIAL, Uri.parse("tel:${Uri.encode(phone)}")), "dialer for $label")
        return if (dialResult.startsWith("Could not")) {
            "CALL_PHONE permission is missing and I could not open the dialer: $dialResult"
        } else {
            "CALL_PHONE permission is missing, so I opened the dialer for ${target.displayName ?: phone}."
        }
    }

    private fun sendSms(recipient: String, message: String): String {
        if (message.isBlank()) return "I need a message before sending SMS."
        val target = resolvePhoneTarget(recipient)
        val phone = target.number
        if (phone.isNullOrBlank()) {
            return if (target.contactPermissionMissing) {
                "READ_CONTACTS permission is required before I can find $recipient. Provide a phone number or grant contacts permission."
            } else {
                "I could not find a phone number for $recipient."
            }
        }

        if (!hasPermission(Manifest.permission.SEND_SMS)) {
            return openSmsComposer(phone, message, "SEND_SMS permission is missing, so I opened the SMS composer instead.")
        }

        return try {
            val smsManager = context.getSystemService(SmsManager::class.java)
            val parts = smsManager.divideMessage(message)
            if (parts.size > 1) {
                smsManager.sendMultipartTextMessage(phone, null, parts, null, null)
            } else {
                smsManager.sendTextMessage(phone, null, message, null, null)
            }
            "SMS sent to ${target.displayName ?: phone}."
        } catch (e: Exception) {
            Log.e(appTag, "Could not send SMS directly", e)
            openSmsComposer(phone, message, "Could not send SMS directly (${e.message}); opened the SMS composer instead.")
        }
    }

    private fun openSmsComposer(phone: String, message: String, prefix: String): String {
        val result = launchIntent(
            Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:${Uri.encode(phone)}"))
                .putExtra("sms_body", message),
            "SMS composer"
        )
        return if (result.startsWith("Could not")) "$prefix $result" else prefix
    }

    private fun playYoutubeMusic(query: String): String {
        if (query.isBlank()) return "I need a song, artist, playlist, or genre to play."
        val encoded = Uri.encode(query)
        val intents = listOf(
            Intent(Intent.ACTION_VIEW, Uri.parse("vnd.youtube://results?search_query=$encoded")).setPackage("com.google.android.youtube"),
            Intent(Intent.ACTION_VIEW, Uri.parse("https://music.youtube.com/search?q=$encoded")),
            Intent(Intent.ACTION_VIEW, Uri.parse("https://www.youtube.com/results?search_query=$encoded"))
        )
        intents.forEach { intent ->
            val result = launchIntent(intent, "YouTube music search for $query")
            if (!result.startsWith("Could not")) return result
        }
        return "Could not start YouTube or YouTube Music."
    }

    private fun openRokidTranslator(): String {
        val packageNames = listOf("com.rokid.translate", "com.rokid.translator", "com.rokid.glass.translator", "com.rokid.ai.translate")
        packageNames.forEach { packageName ->
            context.packageManager.getLaunchIntentForPackage(packageName)?.let { return launchIntent(it, "Rokid real-time translator") }
        }
        return "I could not find the Rokid translator app on this phone."
    }

    @SuppressLint("MissingPermission")
    private fun getGpsLocation(): String {
        if (!hasPermission(Manifest.permission.ACCESS_FINE_LOCATION) && !hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)) {
            return "Location permission is required before I can read GPS hands-free."
        }
        return try {
            val manager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER)
            val location = providers.asSequence().mapNotNull { provider ->
                runCatching { manager.getLastKnownLocation(provider) }.getOrNull()
            }.maxByOrNull { it.time }
            if (location == null) "No last known location is available yet." else "Current location: latitude=${location.latitude}, longitude=${location.longitude}, accuracy=${location.accuracy}m."
        } catch (e: Exception) {
            "Could not read location: ${e.message}"
        }
    }

    private fun getWeather(location: String): String {
        val query = if (location.isNotBlank()) location else getGpsLocation().takeIf { it.startsWith("Current location:") }?.let {
            val lat = Regex("latitude=([-0-9.]+)").find(it)?.groupValues?.getOrNull(1)
            val lon = Regex("longitude=([-0-9.]+)").find(it)?.groupValues?.getOrNull(1)
            if (lat != null && lon != null) "$lat,$lon" else null
        }.orEmpty()
        if (query.isBlank()) return "I need a location or GPS permission to fetch weather."
        return fetchText("https://wttr.in/${URLEncoder.encode(query, "UTF-8")}?format=3", "weather")
    }

    private fun webSearch(query: String): String {
        if (query.isBlank()) return "I need a query to search the web."
        val url = "https://api.duckduckgo.com/?q=${URLEncoder.encode(query, "UTF-8")}&format=json&no_html=1&skip_disambig=1"
        val raw = fetchText(url, "web search")
        val parsed = runCatching { gson.fromJson(raw, Map::class.java) as Map<*, *> }.getOrNull() ?: return raw.take(800)
        val abstract = parsed["AbstractText"]?.toString().orEmpty()
        val heading = parsed["Heading"]?.toString().orEmpty()
        val related = (parsed["RelatedTopics"] as? List<*>)?.firstOrNull()?.let { it as? Map<*, *> }?.get("Text")?.toString().orEmpty()
        return listOf(heading, abstract, related).filter { it.isNotBlank() }.joinToString("\n").ifBlank { "No instant answer found for $query." }
    }

    private fun fetchText(url: String, label: String): String = try {
        val request = Request.Builder().url(url).get().build()
        getClient().newCall(request).execute().use { response ->
            if (!response.isSuccessful) "Could not fetch $label: HTTP ${response.code}" else response.body?.string()?.trim().orEmpty()
        }
    } catch (e: Exception) {
        "Could not fetch $label: ${e.message}"
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
    fun callOpenAIStreaming(
        instruction: String,
        image: Bitmap?,
        allowGlassesPhotoTool: Boolean = false,
        includePersistedHistory: Boolean = true,
        persistConversation: Boolean = true,
        conversationHistory: List<Message> = emptyList()
    ) {
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

                val requestChatId = if (persistConversation) activeChatId() else null

                // Build the messages list with system prompt, persisted chat history, and current user turn.
                val messagesList = mutableListOf<Message>()

                // Add system message
                val sessionPrompt = if (persistConversation) {
                    "Du hast persistente, mehrere Chat-Verläufe. Nutze new_chat, list_chats, switch_chat und rename_chat nur, wenn der Nutzer neue Chats, getrennte Themen, Chatlisten, Umbenennungen oder Chatwechsel wünscht."
                } else if (conversationHistory.isNotEmpty()) {
                    "Du hast nur den Verlauf dieser gerade aktiven Brillen-Konversation. Verwende keine früheren Tests, Kontakte, Telefonnummern, Pläne oder Tool-Ergebnisse aus anderen Konversationen."
                } else {
                    "Diese Anfrage ist eigenständig und hat keinen Chat-Verlauf. Verwende keine früheren Tests, Kontakte, Telefonnummern, Pläne oder Tool-Ergebnisse, wenn sie nicht im aktuellen Nutzertext stehen."
                }
                val systemPrompt = getSystemPrompt() + "\n\nAntworte standardmäßig auf Deutsch (Österreich), knapp und freihändig nutzbar. Nutze Tools nur bei klarer Handlungsabsicht des Nutzers; kurze Proben wie 'test', 'ping' oder 'hallo' sind nur Gespräch und dürfen keine Apps öffnen, keine Suche starten, keine Übersetzer-App öffnen und kein Foto auslösen. Wenn der Nutzer stoppen, abbrechen oder die Konversation beenden will, nutze stop_conversation. Bei Anruf oder SMS gilt: Ein Kontaktname ist ein vollständiger Empfänger. Frage nicht nach der Telefonnummer, wenn ein Name genannt wurde; rufe place_phone_call oder send_sms mit dem Namen auf und lass die App Kontakte auflösen. Frage nur nach, wenn Empfänger oder Nachricht wirklich fehlen. Bei Navigation, Kalender oder E-Mail frage nur nach fehlenden Pflichtangaben. $sessionPrompt"
                messagesList.add(
                    Message(
                        role = "system",
                        content = listOf(Content(type = "text", text = systemPrompt))
                    )
                )

                // Add persisted chat history before the current turn. Image content is not persisted.
                // Probe utterances are intentionally isolated so "test" cannot inherit an old action context.
                if (includePersistedHistory && !isProbeInstruction(instruction)) {
                    messagesList.addAll(activeChatHistoryMessages())
                }
                if (conversationHistory.isNotEmpty() && !isProbeInstruction(instruction)) {
                    messagesList.addAll(conversationHistory.takeLast(20))
                }

                // Add user message
                messagesList.add(
                    Message(
                        role = "user",
                        content = contentList
                    )
                )

                val toolResolution = if (shouldResolveTools(instruction)) {
                    resolveToolCallsIfNeeded(
                        messagesList,
                        instruction,
                        allowGlassesPhotoTool,
                        includeChatTools = persistConversation
                    )
                } else {
                    Log.d(appTag, "Skipping tool planning for non-action instruction: $instruction")
                    ToolResolutionResult(messagesList, emptyList())
                }
                val toolMessages = toolResolution.messages.toMutableList()
                if (toolResolution.results.isNotEmpty()) {
                    toolMessages.add(
                        Message(
                            role = "user",
                            content = "Gib jetzt eine knappe deutschsprachige Erfolgsmeldung oder Fehlermeldung zu diesen Tool-Ergebnissen aus. Frage nicht erneut nach bereits aufgelösten Daten:\n${toolResolution.results.joinToString("\n")}"
                        )
                    )
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
                getClient(isStreaming = true).newCall(httpRequest).execute().use { response ->
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
                        if (persistConversation && requestChatId != null) {
                            appendChatMessages(
                                activeChatId().takeIf { it != requestChatId } ?: requestChatId,
                                listOf(
                                    PersistedChatMessage("user", instruction),
                                    PersistedChatMessage("assistant", finalResponse)
                                )
                            )
                        }
                        listener?.onOpenAIResponse(finalResponse)
                    } else {
                        val fallbackResponse = buildToolResultFallback(toolResolution.results)
                        if (fallbackResponse.isNotBlank()) {
                            Log.w(appTag, "Streaming completed empty after tool call; using fallback response: $fallbackResponse")
                            listener?.onOpenAIStreamingChunk(fallbackResponse, true)
                            if (persistConversation && requestChatId != null) {
                                appendChatMessages(
                                    activeChatId().takeIf { it != requestChatId } ?: requestChatId,
                                    listOf(
                                        PersistedChatMessage("user", instruction),
                                        PersistedChatMessage("assistant", fallbackResponse)
                                    )
                                )
                            }
                            listener?.onOpenAIResponse(fallbackResponse)
                        } else {
                            Log.w(appTag, "Streaming completed but no content was received")
                            listener?.onOpenAIFailed("Empty response")
                        }
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


    private fun resolveToolCallsIfNeeded(
        messagesList: MutableList<Message>,
        instruction: String,
        allowGlassesPhotoTool: Boolean,
        includeChatTools: Boolean
    ): ToolResolutionResult {
        val tools = assistantTools(instruction, allowGlassesPhotoTool, includeChatTools)
        val toolResults = mutableListOf<String>()

        return try {
            repeat(MAX_TOOL_ROUNDS) { round ->
                val request = OpenAIRequest(
                    model = getVlmModel(),
                    messages = messagesList,
                    maxTokens = getVlmMaxTokens(),
                    stream = false,
                    tools = tools,
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

                getClient().newCall(httpRequest).execute().use { response ->
                    if (!response.isSuccessful) {
                        Log.w(appTag, "Tool planning skipped: ${response.code} - ${response.body?.string()}")
                        return ToolResolutionResult(messagesList, toolResults)
                    }

                    val responseBody = response.body?.string() ?: return ToolResolutionResult(messagesList, toolResults)
                    val chatResponse = gson.fromJson(responseBody, ChatCompletionResponse::class.java)
                    val assistantMessage = chatResponse.choices.firstOrNull()?.message ?: return ToolResolutionResult(messagesList, toolResults)
                    val toolCalls = assistantMessage.toolCalls.orEmpty()
                    messagesList.add(assistantMessage)

                    if (toolCalls.isEmpty()) return ToolResolutionResult(messagesList, toolResults)

                    Log.d(appTag, "Executing ${toolCalls.size} tool call(s), round ${round + 1}")
                    toolCalls.forEach { toolCall ->
                        val result = executeAssistantTool(toolCall)
                        toolResults += "${toolCall.function.name}: $result"
                        listener?.onAssistantToolResult(toolCall.function.name, result)
                        messagesList.add(
                            Message(
                                role = "tool",
                                content = result,
                                toolCallId = toolCall.id
                            )
                        )
                    }

                    if (toolCalls.any { it.function.name == "stop_conversation" }) return ToolResolutionResult(messagesList, toolResults)
                }
            }
            Log.w(appTag, "Tool planning reached max rounds ($MAX_TOOL_ROUNDS); continuing with accumulated tool results")
            ToolResolutionResult(messagesList, toolResults)
        } catch (e: Exception) {
            Log.e(appTag, "Tool planning failed; continuing without tools", e)
            ToolResolutionResult(messagesList, toolResults)
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

        val ttsInput = prepareTtsInput(text)
        if (ttsInput.isBlank()) {
            listener?.onTtsFailed("No speakable text")
            return
        }

        Thread {
            try {
                // Create the TTS request
                val request = TtsRequest(
                    model = getTtsModel(),
                    input = ttsInput,
                    voice = getTtsVoice(),
                    responseFormat = "wav"
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

    private fun prepareTtsInput(text: String): String {
        val normalized = text
            .replace(Regex("\\s+"), " ")
            .trim()
        if (normalized.length <= MAX_TTS_INPUT_CHARS) return normalized

        val clipped = normalized.take(MAX_TTS_INPUT_CHARS)
        val sentenceEnd = listOf(
            clipped.lastIndexOf('.'),
            clipped.lastIndexOf('!'),
            clipped.lastIndexOf('?')
        ).maxOrNull() ?: -1
        val wordEnd = clipped.lastIndexOf(' ')
        val cutIndex = when {
            sentenceEnd >= 80 -> sentenceEnd + 1
            wordEnd >= 80 -> wordEnd
            else -> clipped.length
        }
        return clipped.take(cutIndex).trim().ifBlank { clipped.trim() }
    }

    /**
     * Call TTS API in streaming mode
     */
    private fun callTtsAPIStreaming(httpRequest: Request, outputDir: File) {
        // Execute the request with streaming
        getClient(isStreaming = true).newCall(httpRequest).execute().use { response ->
            val contentType = response.body?.contentType()
            val contentLength = response.body?.contentLength()
            Log.d(appTag, "TTS response received: code=${response.code}, contentType=$contentType, contentLength=$contentLength")

            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                Log.e(appTag, "TTS API call failed: ${response.code} - $errorBody")
                listener?.onTtsFailed("TTS API call failed: ${response.code} ${errorBody.take(160)}")
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
            val audioFile = File(outputDir, "tts_result_$timestamp.wav")
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
        getClient(isStreaming = true).newCall(httpRequest).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                Log.e(appTag, "TTS API call failed: ${response.code} - $errorBody")
                listener?.onTtsFailed("TTS API call failed: ${response.code} ${errorBody.take(160)}")
                return
            }

            val responseBody = response.body
            if (responseBody == null) {
                Log.e(appTag, "Empty TTS response body")
                listener?.onTtsFailed("Empty audio response")
                return
            }

            val contentType = responseBody.contentType()?.toString().orEmpty()
            if (contentType.isNotBlank() && !contentType.startsWith("audio/")) {
                val responseText = responseBody.string()
                Log.e(appTag, "TTS API returned non-audio content: $contentType - $responseText")
                listener?.onTtsFailed("TTS API returned $contentType: ${responseText.take(160)}")
                return
            }

            // Save the audio data to file
            val audioBytes = responseBody.bytes()
            if (audioBytes.size > MIN_TTS_AUDIO_BYTES) {
                val timestamp = System.currentTimeMillis()
                val audioFile = File(outputDir, "tts_result_$timestamp.wav")

                audioFile.outputStream().use { fileOut ->
                    fileOut.write(audioBytes)
                }

                Log.d(appTag, "TTS audio saved to: ${audioFile.absolutePath} (${audioBytes.size} bytes)")
                listener?.onTtsComplete(audioFile)
            } else {
                Log.e(appTag, "TTS response body too small: ${audioBytes.size} bytes")
                listener?.onTtsFailed("Invalid audio response: ${audioBytes.size} bytes")
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
