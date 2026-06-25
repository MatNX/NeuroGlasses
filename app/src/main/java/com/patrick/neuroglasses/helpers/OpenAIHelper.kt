package com.patrick.neuroglasses.helpers

import android.annotation.SuppressLint
import android.Manifest
import android.app.Activity
import android.app.ActivityOptions
import android.app.AlarmManager
import android.app.SearchManager
import android.content.Context
import android.content.Intent
import android.app.KeyguardManager
import android.app.PendingIntent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.AlarmClock
import android.provider.CalendarContract
import android.provider.MediaStore
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
import com.google.gson.JsonElement
import com.google.gson.annotations.SerializedName
import com.patrick.neuroglasses.activities.SettingsActivity
import com.patrick.neuroglasses.receivers.ReminderReceiver
import com.patrick.neuroglasses.services.RokidConnectionService
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.util.concurrent.TimeUnit
import java.net.URLEncoder
import java.text.DateFormat
import java.util.Date
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

data class PersistentConversationSummary(
    val id: String,
    val title: String,
    val createdAt: Long,
    val updatedAt: Long,
    val messageCount: Int,
    val isActive: Boolean
)

/**
 * Groq AI Helper
 * Handles AI-related API calls including ASR (speech-to-text) and OpenAI chat completion
 */
class OpenAIHelper(private val context: Context, private val appTag: String = "OpenAIHelper") {

    private val gson = Gson()
    private val chatPrefs by lazy { context.getSharedPreferences("assistant_chats", Context.MODE_PRIVATE) }
    private val navigationDestinationResolver by lazy { NavigationDestinationResolver(context, appTag) }
    @Volatile private var activeAsrCall: Call? = null

    companion object {
        const val DEFERRED_PHOTO_TOOL_RESULT = "__neuroglasses_deferred_photo__"
        const val LOCAL_TOOL_HANDLED_RESULT = "__neuroglasses_local_tool_handled__"

        private const val MAX_TOOL_ROUNDS = 3
        private const val TOOL_PLANNING_MAX_TOKENS = 256
        private const val TOOL_PLANNING_TIMEOUT_SECONDS = 8L
        private const val WEB_FETCH_TIMEOUT_SECONDS = 5L
        private const val STREAMING_READ_TIMEOUT_SECONDS = 35L
        private const val MAX_TTS_INPUT_CHARS = 200
        private const val MIN_TTS_AUDIO_BYTES = 44
        private const val MIN_ASR_TIMEOUT_SECONDS = 30L
        private const val ROKID_SPRITE_LAUNCHER_PACKAGE = "com.rokid.os.sprite.launcher"
        private const val PREF_ACTIVE_CHAT_ID = "active_chat_id"
        private const val PREF_ACTIVE_CHAT_EXPLICIT = "active_chat_explicit"
        private const val WEB_SEARCH_MAX_OUTPUT_CHARS = 5000
        private const val WEB_SEARCH_USER_AGENT =
            "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) NeuroGlasses/1.0 Mobile Safari/537.36"
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
        val results: List<String>,
        val directResponse: String? = null,
        val deferred: Boolean = false
    )

    private data class ChatRequestContext(
        val request: OpenAIRequest,
        val toolResults: List<String>,
        val requestChatId: String?,
        val hasImage: Boolean,
        val directResponse: String? = null,
        val deferred: Boolean = false
    )

    private data class LocationCoordinates(
        val latitude: Double,
        val longitude: Double,
        val accuracyMeters: Float
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
        val readTimeoutSeconds = if (isStreaming) {
            minOf(timeoutSeconds * 2, STREAMING_READ_TIMEOUT_SECONDS)
        } else {
            timeoutSeconds
        }
        return OkHttpClient.Builder()
            .connectTimeout(timeoutSeconds, TimeUnit.SECONDS)
            // For streaming, use a longer read timeout to allow for slower chunk delivery
            // but still respect the configured timeout as a baseline
            .readTimeout(readTimeoutSeconds, TimeUnit.SECONDS)
            .writeTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .build()
    }

    private fun getBoundedClient(timeoutSeconds: Long): OkHttpClient {
        val configuredTimeout = getApiTimeout().toLong().coerceAtLeast(1L)
        val boundedTimeout = minOf(configuredTimeout, timeoutSeconds)
        return getClient()
            .newBuilder()
            .callTimeout(boundedTimeout, TimeUnit.SECONDS)
            .connectTimeout(boundedTimeout, TimeUnit.SECONDS)
            .readTimeout(boundedTimeout, TimeUnit.SECONDS)
            .writeTimeout(boundedTimeout, TimeUnit.SECONDS)
            .build()
    }

    private fun getToolPlanningClient(): OkHttpClient =
        getBoundedClient(TOOL_PLANNING_TIMEOUT_SECONDS)

    private fun getWebFetchClient(): OkHttpClient =
        getBoundedClient(WEB_FETCH_TIMEOUT_SECONDS)

    private fun getChatApiUrl(): String = "${getApiBaseUrl()}/chat/completions"
    private fun getAsrApiUrl(): String = "${getApiBaseUrl()}/audio/transcriptions"
    private fun getTtsApiUrl(): String = "${getApiBaseUrl()}/audio/speech"


    private fun getChatSessions(): MutableList<PersistedChatSession> {
        val json = chatPrefs.getString("sessions", null) ?: return mutableListOf()
        return runCatching {
            gson.fromJson(json, Array<PersistedChatSession>::class.java).toMutableList()
        }.getOrElse {
            Log.w(appTag, "Could not read persisted chats", it)
            mutableListOf()
        }
    }

    private fun saveChatSessions(sessions: List<PersistedChatSession>) {
        chatPrefs.edit().putString("sessions", gson.toJson(sessions)).apply()
    }

    private fun activeChatIdOrNull(): String? {
        val sessions = getChatSessions()
        val explicit = chatPrefs.getBoolean(PREF_ACTIVE_CHAT_EXPLICIT, false)
        val stored = chatPrefs.getString(PREF_ACTIVE_CHAT_ID, null)
        if (!explicit) {
            if (stored != null) chatPrefs.edit().remove(PREF_ACTIVE_CHAT_ID).apply()
            return null
        }
        if (stored != null && sessions.any { it.id == stored }) return stored
        if (stored != null) {
            chatPrefs.edit()
                .remove(PREF_ACTIVE_CHAT_ID)
                .remove(PREF_ACTIVE_CHAT_EXPLICIT)
                .apply()
        }
        return null
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
        chatPrefs.edit()
            .putString(PREF_ACTIVE_CHAT_ID, session.id)
            .putBoolean(PREF_ACTIVE_CHAT_EXPLICIT, true)
            .apply()
        return session
    }

    fun listPersistentConversations(): List<PersistentConversationSummary> {
        val active = activeChatIdOrNull()
        return getChatSessions().map { session ->
            PersistentConversationSummary(
                id = session.id,
                title = session.title,
                createdAt = session.createdAt,
                updatedAt = session.updatedAt,
                messageCount = session.messages.size,
                isActive = session.id == active
            )
        }
    }

    fun hasActivePersistentConversation(): Boolean = activeChatIdOrNull() != null

    fun activePersistentConversationId(): String? = activeChatIdOrNull()

    fun createPersistentConversation(title: String? = null): PersistentConversationSummary =
        createChatSession(title).toSummary(isActive = true)

    fun enterPersistentConversation(chatId: String): Boolean {
        val exists = getChatSessions().any { it.id == chatId }
        if (!exists) return false
        chatPrefs.edit()
            .putString(PREF_ACTIVE_CHAT_ID, chatId)
            .putBoolean(PREF_ACTIVE_CHAT_EXPLICIT, true)
            .apply()
        return true
    }

    fun leavePersistentConversation() {
        chatPrefs.edit()
            .remove(PREF_ACTIVE_CHAT_ID)
            .remove(PREF_ACTIVE_CHAT_EXPLICIT)
            .apply()
    }

    fun renamePersistentConversation(chatId: String?, title: String): Boolean {
        if (title.isBlank()) return false
        val targetId = chatId?.takeIf { it.isNotBlank() } ?: activeChatIdOrNull() ?: return false
        return updateChatSession(targetId) { it.copy(title = title) } != null
    }

    fun deletePersistentConversation(chatId: String): Boolean {
        if (chatId.isBlank()) return false
        val sessions = getChatSessions()
        val removed = sessions.removeAll { it.id == chatId }
        if (!removed) return false
        saveChatSessions(sessions)
        if (activeChatIdOrNull() == chatId) {
            leavePersistentConversation()
        }
        return true
    }

    private fun PersistedChatSession.toSummary(isActive: Boolean): PersistentConversationSummary =
        PersistentConversationSummary(
            id = id,
            title = title,
            createdAt = createdAt,
            updatedAt = updatedAt,
            messageCount = messages.size,
            isActive = isActive
        )

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

    private fun chatHistoryMessages(chatId: String): List<Message> {
        val session = getChatSessions().firstOrNull { it.id == chatId } ?: return emptyList()
        return session.messages.takeLast(20).map { Message(role = it.role, content = it.text) }
    }

    private fun switchChat(chatId: String?, title: String?): String {
        val sessions = getChatSessions()
        val session = when {
            !chatId.isNullOrBlank() -> sessions.firstOrNull { it.id == chatId }
            !title.isNullOrBlank() -> sessions.firstOrNull { it.title.equals(title, ignoreCase = true) }
            else -> null
        } ?: return "Chat not found. Use list_chats to see available chat ids and titles."
        chatPrefs.edit()
            .putString(PREF_ACTIVE_CHAT_ID, session.id)
            .putBoolean(PREF_ACTIVE_CHAT_EXPLICIT, true)
            .apply()
        return "Switched to chat '${session.title}' (${session.id})."
    }

    private fun listChats(): String {
        val active = activeChatIdOrNull()
        val sessions = getChatSessions()
        if (sessions.isEmpty()) return "No chats exist yet."
        return sessions.joinToString("\n") { session ->
            val marker = if (session.id == active) "*" else "-"
            "$marker ${session.title} [${session.id}] (${session.messages.size} messages)"
        }
    }

    private fun renameChat(chatId: String?, title: String): String {
        if (title.isBlank()) return "A new chat title is required."
        val targetId = chatId?.takeIf { it.isNotBlank() } ?: activeChatIdOrNull()
            ?: return "No persistent chat is active."
        val updated = updateChatSession(targetId) { it.copy(title = title) }
        return if (updated == null) "Chat not found." else "Renamed chat to '${updated.title}'."
    }

    private fun leaveChat(): String {
        leavePersistentConversation()
        return "Left persistent chat. The main conversation is ephemeral again."
    }

    private fun deleteChat(chatId: String?): String {
        val targetId = chatId?.takeIf { it.isNotBlank() } ?: activeChatIdOrNull()
            ?: return "No persistent chat is active."
        return if (deletePersistentConversation(targetId)) {
            "Deleted persistent chat."
        } else {
            "Chat not found."
        }
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

    private fun replaceActiveAsrCall(call: Call) {
        synchronized(this) {
            activeAsrCall?.cancel()
            activeAsrCall = call
        }
    }

    private fun clearActiveAsrCall(call: Call?) {
        if (call == null) return
        synchronized(this) {
            if (activeAsrCall === call) activeAsrCall = null
        }
    }

    private fun isActiveAsrCall(call: Call?): Boolean =
        call != null && synchronized(this) { activeAsrCall === call }

    private fun notifyAsrCompleteIfActive(call: Call?, text: String) {
        if (isActiveAsrCall(call)) {
            listener?.onAsrComplete(text)
        } else {
            Log.d(appTag, "Ignoring stale ASR completion")
        }
    }

    private fun notifyAsrFailedIfActive(call: Call?, error: String) {
        if (isActiveAsrCall(call)) {
            listener?.onAsrFailed(error)
        } else {
            Log.d(appTag, "Ignoring stale ASR failure: $error")
        }
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
            var call: Call? = null
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

                // Execute the request with a total call timeout so ASR cannot hang silently.
                val requestCall = getAsrClient().newCall(request)
                call = requestCall
                replaceActiveAsrCall(requestCall)
                requestCall.execute().use { response ->
                    if (!response.isSuccessful) {
                        val errorBody = response.body?.string() ?: "Unknown error"
                        Log.e(appTag, "ASR API call failed: ${response.code} - $errorBody")
                        notifyAsrFailedIfActive(call, "ASR API call failed: ${response.code}")
                        return@Thread
                    }

                    val responseBody = response.body?.string()
                    Log.d(appTag, "ASR Response: $responseBody")

                    if (responseBody != null) {
                        val asrResponse = gson.fromJson(responseBody, AsrResponse::class.java)
                        val recognizedText = asrResponse.text

                        if (recognizedText.isNotEmpty()) {
                            Log.d(appTag, "ASR recognized text: $recognizedText")
                            notifyAsrCompleteIfActive(call, recognizedText)
                        } else {
                            Log.e(appTag, "ASR returned empty text")
                            notifyAsrFailedIfActive(call, "No text recognized")
                        }
                    } else {
                        Log.e(appTag, "Empty ASR response body")
                        notifyAsrFailedIfActive(call, "Empty response")
                    }
                }
            } catch (e: IOException) {
                Log.e(appTag, "Network error during ASR: ${e.message}", e)
                if (call?.isCanceled() != true) {
                    notifyAsrFailedIfActive(call, "Network error: ${e.message}")
                }
            } catch (e: Exception) {
                Log.e(appTag, "Error calling ASR API: ${e.message}", e)
                notifyAsrFailedIfActive(call, "Error: ${e.message}")
            } finally {
                clearActiveAsrCall(call)
            }
        }.start()
    }

    fun cancelAsrRequest(reason: String = "ASR cancelled") {
        Log.d(appTag, "Cancelling ASR request: $reason")
        synchronized(this) {
            activeAsrCall?.cancel()
            activeAsrCall = null
        }
    }

    private fun getAsrClient(): OkHttpClient {
        val timeoutSeconds = maxOf(getApiTimeout().toLong(), MIN_ASR_TIMEOUT_SECONDS)
        return getClient()
            .newBuilder()
            .callTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .connectTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .readTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .writeTimeout(timeoutSeconds, TimeUnit.SECONDS)
            .build()
    }


    private fun assistantTools(
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
        fun numericStringProp(description: String): Map<String, Any> = mapOf("type" to "string", "description" to description)

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
                description = "Close the active NeuroGlasses conversation when the user clearly wants to stop, cancel, say goodbye, or end the session. Do not use this for ordinary conversation.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "open_rokid_native_app",
                description = "Open a native/glasses-side Rokid app through Hi Rokid CXR-L. Use this instead of phone apps for Rokid translation, camera, gallery, settings, brightness, volume, teleprompter, music/lyrics, recorder, or other glasses-native requests. For navigation with a destination, use start_navigation so NeuroGlasses can resolve and confirm the address first.",
                parameters = objectSchema(listOf("app"), mapOf(
                    "app" to mapOf(
                        "type" to "string",
                        "description" to "Native glasses app to open.",
                        "enum" to listOf(
                            "translator",
                            "navigation",
                            "camera",
                            "gallery",
                            "settings",
                            "bluetooth_settings",
                            "system_info",
                            "brightness",
                            "volume",
                            "teleprompter",
                            "music",
                            "recorder",
                            "browser",
                            "launcher",
                            "manager"
                        )
                    )
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "start_navigation",
                description = "Resolve an Austrian destination before starting NeuroGlasses background navigation. Use this for navigation, maps, routes, or directions. The tool may return candidate locations and ask the user to confirm one by number. Default mode is public_transit; use foot only when the user explicitly asks to walk or go by foot.",
                parameters = objectSchema(listOf("destination"), mapOf(
                    "destination" to stringProp("Best transcript of the destination/address, preserving all heard German street/place words, house numbers, postcodes, and city names."),
                    "mode" to mapOf(
                        "type" to "string",
                        "description" to "Route preference. Default public_transit; use foot for explicit walking/by-foot requests.",
                        "enum" to listOf("public_transit", "foot")
                    )
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "confirm_navigation_destination",
                description = "Confirm or refine a pending navigation destination after start_navigation returned candidate options. Use this when the user says 'eins', 'die zweite', 'ja die erste', a candidate name, or gives a corrected address. If corrected, pass the corrected text as destination.",
                parameters = objectSchema(listOf("selection"), mapOf(
                    "selection" to stringProp("The user's choice such as '1', 'eins', 'zweite', a candidate name, 'nein', or the corrected spoken text."),
                    "destination" to stringProp("Optional corrected full destination/address if the user did not choose one of the numbered options."),
                    "mode" to mapOf(
                        "type" to "string",
                        "description" to "Optional route preference to keep or override.",
                        "enum" to listOf("public_transit", "foot")
                    )
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "stop_navigation",
                description = "Stop the active NeuroGlasses background navigation session when the user asks to end, cancel, or stop navigation.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "play_youtube_music",
                description = "Play requested music/video hands-free. If you know a direct YouTube URL, pass video_url. Otherwise pass query. The app will try Android media playback first for background-capable apps, then fall back to browser.",
                parameters = objectSchema(listOf("query"), mapOf(
                    "query" to stringProp("Song, artist, playlist, video title, or search query to play."),
                    "video_url" to stringProp("Optional direct YouTube URL if known.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "open_rokid_translator",
                description = "Open the native Rokid/glasses-side real-time translator or translation app through Hi Rokid CXR-L. Use this for translate/übersetzen requests, not Google Translate on the phone.",
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
                description = "Use DuckDuckGo instant answers for current/web questions. If no structured instant answer exists, open a visible web search for the user instead of scraping result pages.",
                parameters = objectSchema(listOf("query"), mapOf("query" to stringProp("Search query.")))
            )),
            AssistantTool(function = ToolFunction(
                name = "set_timer",
                description = "Create a hands-free Android timer. Convert the user's requested duration yourself and pass a positive seconds value as decimal digits.",
                parameters = objectSchema(listOf("seconds"), mapOf(
                    "seconds" to numericStringProp("Timer duration in seconds, as decimal digits. Must be greater than zero."),
                    "label" to stringProp("Optional timer label.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "create_reminder",
                description = "Create a reminder. If the user gives a due time, calculate the exact local epoch milliseconds yourself and pass due_at_epoch_millis as decimal digits; the app will not interpret natural time text.",
                parameters = objectSchema(listOf("text"), mapOf(
                    "text" to stringProp("Reminder text."),
                    "when" to stringProp("Optional human-readable due time for display only."),
                    "due_at_epoch_millis" to numericStringProp("Optional exact local due time as Unix epoch milliseconds, as decimal digits.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "list_reminders",
                description = "Read reminders previously stored by the assistant.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "create_calendar_event",
                description = "Deutschsprachig: Lege einen Kalendertermin per Sprache an. Berechne Start/Ende selbst und uebergib epoch milliseconds als Dezimalziffern, wenn der Nutzer Zeiten nennt.",
                parameters = objectSchema(listOf("title"), mapOf(
                    "title" to stringProp("Titel des Termins."),
                    "start" to stringProp("Optionale Startzeit als lesbarer Text fuer die Beschreibung."),
                    "end" to stringProp("Optionale Endzeit als lesbarer Text fuer die Beschreibung."),
                    "start_epoch_millis" to numericStringProp("Optionaler exakter Startzeitpunkt als Unix epoch milliseconds, als Dezimalziffern."),
                    "end_epoch_millis" to numericStringProp("Optionaler exakter Endzeitpunkt als Unix epoch milliseconds, als Dezimalziffern."),
                    "location" to stringProp("Optionaler Ort."),
                    "notes" to stringProp("Optionale Notizen.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = "send_email",
                description = "Deutschsprachig: Öffne Gmail oder die Standard-Mail-App mit Empfänger, Betreff und Text. Android/Gmail verlangt Nutzerbestätigung zum Senden; dieses Tool verschickt nicht heimlich im Hintergrund.",
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
                description = "Capture a fresh Rokid glasses camera photo as visual context. Use this whenever the user asks about the current view, scene, objects, people, hands, counts, colors, visible text, translation of visible text, or anything that needs eyesight. Do not answer visual questions from memory when no image is attached.",
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
            )),
            AssistantTool(function = ToolFunction(
                name = "leave_chat",
                description = "Leave the active persistent chat and return to the ephemeral main conversation when the user asks to stop using saved chat history.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = "delete_chat",
                description = "Delete a persistent chat only when the user explicitly asks to remove it.",
                parameters = objectSchema(emptyList(), mapOf(
                    "chat_id" to stringProp("Optional chat id; omit to delete the active persistent chat.")
                ))
            ))
        )

        return tools.filter { tool ->
            when (tool.function.name) {
                "new_chat", "list_chats", "switch_chat", "rename_chat", "leave_chat", "delete_chat" -> includeChatTools
                "snap_glasses_photo" -> allowGlassesPhotoTool
                else -> true
            }
        }
    }

    private fun executeAssistantTool(toolCall: ToolCall, instruction: String): String {
        val args = decodeToolArguments(toolCall.function.arguments)
        listener?.onAssistantToolCall(toolCall.function.name, args)?.let { return it }

        fun arg(name: String): String = args[name].orEmpty()

        return when (toolCall.function.name) {
            "place_phone_call" -> placePhoneCall(arg("recipient"))
            "send_sms" -> sendSms(arg("recipient"), arg("message"))
            "find_contact_phone" -> findContactPhone(arg("recipient"))
            "stop_conversation" -> "Conversation stopped."
            "open_rokid_native_app" -> openRokidNativeApp(arg("app"))
            "start_navigation" -> startOsmNavigation(arg("destination"), arg("mode"))
            "confirm_navigation_destination" -> confirmOsmNavigationDestination(arg("selection"), arg("destination"), arg("mode"))
            "stop_navigation" -> stopOsmNavigation()
            "play_youtube_music" -> playYoutubeMusic(arg("query"), arg("video_url"))
            "open_rokid_translator" -> openRokidTranslator()
            "get_gps_location" -> getGpsLocation()
            "get_weather" -> getWeather(arg("location"))
            "web_search" -> webSearch(arg("query"))
            "set_timer" -> setTimer(arg("seconds").toIntOrNull() ?: 0, arg("label").ifBlank { instruction })
            "create_reminder" -> createReminder(arg("text"), arg("when"), arg("due_at_epoch_millis").toLongOrNull())
            "list_reminders" -> listReminders()
            "create_calendar_event" -> createCalendarEvent(
                arg("title"),
                arg("start"),
                arg("end"),
                arg("start_epoch_millis").toLongOrNull(),
                arg("end_epoch_millis").toLongOrNull(),
                arg("location"),
                arg("notes")
            )
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
            "leave_chat" -> leaveChat()
            "delete_chat" -> deleteChat(arg("chat_id"))
            else -> "Unknown tool: ${toolCall.function.name}"
        }
    }

    private fun decodeToolArguments(arguments: String): Map<String, String> {
        val decoded = runCatching { gson.fromJson(arguments, JsonElement::class.java) }.getOrNull()
        val jsonObject = decoded?.takeIf { it.isJsonObject }?.asJsonObject ?: return emptyMap()
        return jsonObject.entrySet().associate { (key, value) ->
            key to jsonArgumentToString(value)
        }
    }

    private fun jsonArgumentToString(value: JsonElement?): String {
        if (value == null || value.isJsonNull) return ""
        if (!value.isJsonPrimitive) return value.toString()
        val primitive = value.asJsonPrimitive
        return when {
            primitive.isNumber -> primitive.asBigDecimal.toPlainString()
            primitive.isBoolean -> primitive.asBoolean.toString()
            else -> primitive.asString
        }
    }

    private fun shouldDeferAfterToolResult(toolName: String, result: String): Boolean =
        toolName in setOf("start_navigation", "confirm_navigation_destination")

    private fun navigationActuallyStarted(result: String): Boolean =
        result.startsWith("Navigation mode started", ignoreCase = true)

    private fun toolResultLooksFailed(result: String): Boolean =
        result.startsWith("Could not", ignoreCase = true) ||
            result.startsWith("I need", ignoreCase = true) ||
            result.contains("permission is required", ignoreCase = true) ||
            result.contains("failed", ignoreCase = true)

    private fun buildToolResultFallback(results: List<String>): String {
        if (results.isEmpty()) return ""
        val last = results.last()
        val toolName = last.substringBefore(":").trim()
        val lastResult = last.substringAfter(":").trim()
        return when {
            toolName == "web_search" ->
                "Ich habe die Websuche ausgeführt, aber danach keinen KI-Antworttext bekommen. Hier sind die Rohfunde:\n${lastResult.take(2600)}"
            lastResult.startsWith("Started ", ignoreCase = true) -> "Erledigt: ${lastResult.removePrefix("Started ").removeSuffix(" hands-free.")} gestartet."
            lastResult.startsWith("Requested ", ignoreCase = true) -> lastResult
            lastResult.startsWith("Could not", ignoreCase = true) -> "Das hat leider nicht geklappt: $lastResult"
            lastResult.contains("permission", ignoreCase = true) -> "Dafür fehlt noch eine Berechtigung: $lastResult"
            else -> lastResult
        }
    }

    private fun buildAssistantFailureResponse(
        reason: String,
        detail: String? = null,
        toolResults: List<String> = emptyList()
    ): String {
        val toolFallback = buildToolResultFallback(toolResults)
        val cleanDetail = detail
            ?.collapseWhitespace()
            ?.trim()
            ?.take(360)
            .orEmpty()

        return buildString {
            append("Ich habe gerade keine normale KI-Antwort bekommen: $reason.")
            if (cleanDetail.isNotBlank()) append(" Details: $cleanDetail")
            if (toolFallback.isNotBlank()) {
                append("\n\nSchon ermittelte Ergebnisse:\n")
                append(toolFallback)
            }
        }.take(3200)
    }

    private fun visionInstruction(instruction: String): String =
        "Beantworte die Nutzerfrage anhand des angehängten Kamerabildes. " +
            "Prüfe das Bild selbst und nutze keine früheren Beschreibungen als Ersatz für das Bild. " +
            "Wenn die Frage nach Anzahl, Farbe, Text, Übersetzung oder Details fragt, antworte direkt aus dem Bild. " +
            "Wenn ein Detail nicht sicher erkennbar ist, sag das knapp. Nutzerfrage: $instruction"

    private fun launchIntent(intent: Intent, label: String): String = try {
        val launchIntent = Intent(intent).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        if (context is Activity) {
            context.startActivity(launchIntent)
            "Started $label hands-free."
        } else {
            launchIntentFromBackground(launchIntent, label)
        }
    } catch (e: Exception) {
        Log.e(appTag, "Could not start $label", e)
        val lockedSuffix = if (isDeviceLocked()) " The phone is locked; unlock it and try again if Android blocked the target app." else ""
        "Could not start $label: ${e.message}.$lockedSuffix"
    }

    private fun launchIntentFromBackground(intent: Intent, label: String): String {
        val options = backgroundActivityLaunchOptions()
        val requestCode = (System.currentTimeMillis() and Int.MAX_VALUE.toLong()).toInt()
        val pendingIntent = PendingIntent.getActivity(
            context,
            requestCode,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            options
        )

        return try {
            pendingIntent.send(context, 0, null, null, null, null, options)
            "Requested $label via Android. If nothing opens, Android blocked this background launch."
        } catch (e: PendingIntent.CanceledException) {
            Log.e(appTag, "Could not send pending intent for $label", e)
            "Could not start $label: ${e.message}."
        } catch (e: Exception) {
            Log.e(appTag, "Could not request $label", e)
            "Could not start $label: ${e.message}."
        }
    }

    private fun backgroundActivityLaunchOptions(): Bundle? {
        val options = ActivityOptions.makeBasic()
        val mode = backgroundActivityStartMode() ?: return options.toBundle()
        listOf(
            "setPendingIntentBackgroundActivityStartMode",
            "setPendingIntentCreatorBackgroundActivityStartMode"
        ).forEach { methodName ->
            runCatching {
                ActivityOptions::class.java
                    .getMethod(methodName, Int::class.javaPrimitiveType)
                    .invoke(options, mode)
            }.onFailure {
                Log.d(appTag, "ActivityOptions.$methodName unavailable on API ${Build.VERSION.SDK_INT}")
            }
        }
        return options.toBundle()
    }

    private fun backgroundActivityStartMode(): Int? =
        listOf(
            "MODE_BACKGROUND_ACTIVITY_START_ALLOW_ALWAYS",
            "MODE_BACKGROUND_ACTIVITY_START_ALLOWED"
        ).firstNotNullOfOrNull { fieldName ->
            runCatching {
                ActivityOptions::class.java.getField(fieldName).getInt(null)
            }.getOrNull()
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
            .map { char -> if (char.isLetterOrDigit() || char.isWhitespace()) char else ' ' }
            .joinToString("")
            .collapseWhitespace()
            .trim()

    private fun String.collapseWhitespace(): String =
        splitToSequence(' ', '\n', '\r', '\t')
            .filter { it.isNotBlank() }
            .joinToString(" ")

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

    private fun playYoutubeMusic(query: String, videoUrl: String): String {
        if (query.isBlank()) return "I need a YouTube video, song, artist, playlist, or search query."
        val encoded = Uri.encode(query)
        val searchUrl = "https://www.youtube.com/results?search_query=$encoded"
        val watchUrl = videoUrl.takeIf { it.startsWith("http://") || it.startsWith("https://") }
        val mediaIntents = youtubeMediaPackages().map { packageName ->
            Intent(MediaStore.INTENT_ACTION_MEDIA_PLAY_FROM_SEARCH)
                .setPackage(packageName)
                .putExtra(SearchManager.QUERY, query)
                .putExtra(MediaStore.EXTRA_MEDIA_FOCUS, "vnd.android.cursor.item/video")
        }
        val intents = buildList {
            watchUrl?.let { add(Intent(Intent.ACTION_VIEW, Uri.parse(it))) }
            addAll(mediaIntents)
            add(Intent(Intent.ACTION_VIEW, Uri.parse(searchUrl)))
        }
        val label = "YouTube playback"
        intents.forEach { intent ->
            val result = launchIntent(intent, label)
            if (!result.startsWith("Could not")) return result
        }
        return "Could not start YouTube video/search for $query."
    }

    private fun youtubeMediaPackages(): List<String> = listOf(
        "com.google.android.apps.youtube.music",
        "com.google.android.youtube"
    )

    private fun openRokidTranslator(): String = openRokidNativeApp("translator")

    private fun startOsmNavigation(destination: String, mode: String): String {
        val cleanMode = mode.ifBlank { "public_transit" }
        val resolution = navigationDestinationResolver.resolve(destination, cleanMode)
        return when (resolution.status) {
            NavigationDestinationResolver.Status.CONFIRMED -> {
                val selected = resolution.selected ?: return resolution.message
                RokidConnectionService.startNavigation(
                    context = context,
                    destination = selected.shortLabel,
                    mode = resolution.mode.ifBlank { cleanMode },
                    latitude = selected.latitude,
                    longitude = selected.longitude
                )
            }
            NavigationDestinationResolver.Status.NEEDS_CONFIRMATION,
            NavigationDestinationResolver.Status.NOT_FOUND,
            NavigationDestinationResolver.Status.ERROR,
            NavigationDestinationResolver.Status.CANCELLED -> resolution.message
        }
    }

    private fun confirmOsmNavigationDestination(selection: String, destination: String, mode: String): String {
        val resolution = navigationDestinationResolver.confirm(
            selection = selection,
            refinedDestination = destination.takeIf { it.isNotBlank() },
            modeOverride = mode.takeIf { it.isNotBlank() }
        )
        return when (resolution.status) {
            NavigationDestinationResolver.Status.CONFIRMED -> {
                val selected = resolution.selected ?: return resolution.message
                RokidConnectionService.startNavigation(
                    context = context,
                    destination = selected.shortLabel,
                    mode = resolution.mode.ifBlank { "public_transit" },
                    latitude = selected.latitude,
                    longitude = selected.longitude
                )
            }
            NavigationDestinationResolver.Status.NEEDS_CONFIRMATION,
            NavigationDestinationResolver.Status.NOT_FOUND,
            NavigationDestinationResolver.Status.ERROR,
            NavigationDestinationResolver.Status.CANCELLED -> resolution.message
        }
    }

    private fun stopOsmNavigation(): String =
        RokidConnectionService.stopNavigation(context)

    private fun openRokidNavigation(destination: String): String {
        val launchResult = RokidHostConnection.openGlassApp(rokidNavigationTargets())
        if (!launchResult.success) return launchResult.detail

        val cleanDestination = destination.trim()
        return if (cleanDestination.isNotBlank()) {
            val commandResult = RokidHostConnection.sendNavigationDestination(cleanDestination)
            if (commandResult.success) {
                "Opened ${launchResult.label} on the glasses and sent destination: $cleanDestination."
            } else {
                "Opened ${launchResult.label} on the glasses, but could not send destination '$cleanDestination': ${commandResult.detail}"
            }
        } else {
            "Opened ${launchResult.label} on the glasses."
        }
    }

    private fun openRokidNativeApp(app: String): String {
        val normalized = app.trim().lowercase(Locale.ROOT)
        val targets = when {
            normalized in setOf("translator", "translate", "translation", "übersetzer", "uebersetzer", "übersetzung", "rt translation", "live translation") ->
                rokidTranslatorTargets()
            normalized in setOf("navigation", "maps", "map", "route", "directions", "navigate", "navi", "smartlife") ->
                rokidNavigationTargets()
            normalized in setOf("camera", "kamera", "photo", "foto") ->
                spriteLauncherTarget("Rokid Camera", ".page.camera.CameraPageActivity") +
                    glassTargets("Rokid Camera", "com.rokid.camera", "com.rokid.glass.camera", "com.rokid.ai.camera")
            normalized in setOf("gallery", "galerie", "photos", "bilder") ->
                spriteLauncherTarget("Rokid Gallery", ".page.gallery.StorageImageShowActivity") +
                    glassTargets("Rokid Gallery", "com.rokid.gallery", "com.rokid.glass.gallery", "com.rokid.photos")
            normalized in setOf("settings", "einstellungen", "manager") ->
                spriteLauncherTarget("Rokid Settings", ".setting.SettingPageActivity") +
                    glassTargets("Rokid Settings", "com.rokid.settings", "com.rokid.glass.settings", "com.android.settings") +
                    glassTargets("Rokid Manager", "com.example.advancedsettingsmanager")
            normalized in setOf("bluetooth_settings", "bluetooth", "bt", "bluetooth settings", "bluetooth-einstellungen") ->
                spriteLauncherTarget("Rokid Bluetooth Settings", ".setting.bluetooth.SettingBluetoothActivity")
            normalized in setOf("system_info", "systeminfo", "system info", "info", "geräteinfo", "geraeteinfo") ->
                spriteLauncherTarget("Rokid System Info", ".setting.info.SettingSystemInfoActivity")
            normalized in setOf("brightness", "helligkeit", "display brightness") ->
                spriteLauncherTarget("Rokid Brightness", ".page.brightness.SettingBrightnessActivity")
            normalized in setOf("volume", "lautstärke", "lautstaerke", "sound", "audio volume") ->
                spriteLauncherTarget("Rokid Volume", ".page.volume.SettingVolumeActivity")
            normalized in setOf("teleprompter", "wordtips", "word tips", "worttipps", "spickzettel") ->
                spriteLauncherTarget("Rokid Word Tips", ".page.wordtips.WordTipsPageActivity")
            normalized in setOf("music", "musik", "lyrics", "songtext", "songtexte") ->
                spriteLauncherTarget("Rokid Music", ".page.music.MusicPageActivity")
            normalized in setOf("recorder", "audio_recorder", "audio recorder", "aufnahme", "audioaufnahme", "recording") ->
                spriteLauncherTarget("Rokid Audio Recorder", ".page.audio.AudioPageActivity")
            normalized in setOf("browser", "web") ->
                glassTargets("Rokid Browser", "com.rokid.browser", "com.rokid.glass.browser") +
                    glassTargets("Dew Browser", "kr.pe.eung.ekweb")
            normalized in setOf("launcher", "home", "start") ->
                spriteLauncherTarget("Rokid Home", ".main.SpriteMainActivity") +
                    glassTargets("Rokid Launcher", "com.rokid.launcher", "com.rokid.glass.launcher")
            else -> emptyList()
        }
        if (targets.isEmpty()) {
            return "Unknown Rokid native app '$app'. Supported: translator, navigation, camera, gallery, settings, bluetooth_settings, system_info, brightness, volume, teleprompter, music, recorder, browser, launcher."
        }

        val result = RokidHostConnection.openGlassApp(targets)
        return if (result.success) {
            "Opened ${result.label} on the glasses."
        } else {
            result.detail
        }
    }

    private fun rokidTranslatorTargets(): List<RokidHostConnection.GlassAppTarget> =
        spriteLauncherTarget("Rokid Translate", ".page.translate.TranslatePageActivity") +
            glassTargets(
            "Rokid Real-Time Translation",
            "com.rokid.translate",
            "com.rokid.translator",
            "com.rokid.glass.translator",
            "com.rokid.ai.translate",
            "com.rokid.voice.translation",
            "com.rokid.translation"
        ) + glassTargets("EK Trans", "kr.pe.eung.ektranslator")

    private fun rokidNavigationTargets(): List<RokidHostConnection.GlassAppTarget> =
        spriteLauncherTarget(
            "Rokid Navigation",
            ".page.navigation.NavigationOverseaPageActivity",
            ".page.navigation.NavigationPageActivity"
        ) + glassTargets(
            "Rokid Navigation",
            "com.rokid.navigation",
            "com.rokid.maps",
            "com.rokid.map",
            "com.rokid.glass.navigation",
            "com.rokid.glass.maps",
            "com.rokid.ai.navigation",
            "com.rokid.ar.navigation"
        ) + glassTargets("Rokid SmartLife", "com.rokidsmartlife") +
            glassTargets("Rokid GMaps", "com.anezium.rokidgmaps.glasses")

    private fun spriteLauncherTarget(label: String, vararg relativeActivityNames: String): List<RokidHostConnection.GlassAppTarget> =
        listOf(
            RokidHostConnection.GlassAppTarget(
                label = label,
                packageName = ROKID_SPRITE_LAUNCHER_PACKAGE,
                activityNames = relativeActivityNames
                    .flatMap { activityNameVariants(ROKID_SPRITE_LAUNCHER_PACKAGE, it) }
                    .distinct()
            )
        )

    private fun glassTargets(label: String, vararg packageNames: String): List<RokidHostConnection.GlassAppTarget> =
        packageNames.distinct().map { packageName ->
            RokidHostConnection.GlassAppTarget(
                label = label,
                packageName = packageName,
                activityNames = listOf("", "$packageName.MainActivity", "$packageName.main.MainActivity")
            )
        }

    private fun activityNameVariants(packageName: String, activityName: String): List<String> {
        val trimmed = activityName.trim()
        if (trimmed.isBlank()) return emptyList()
        return if (trimmed.startsWith(".")) {
            listOf(packageName + trimmed, trimmed)
        } else {
            listOf(trimmed)
        }
    }

    @SuppressLint("MissingPermission")
    private fun readLastKnownLocation(): LocationCoordinates? {
        if (!hasPermission(Manifest.permission.ACCESS_FINE_LOCATION) && !hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)) {
            return null
        }
        return runCatching {
            val manager = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
            val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER, LocationManager.PASSIVE_PROVIDER)
            providers.asSequence().mapNotNull { provider ->
                runCatching { manager.getLastKnownLocation(provider) }.getOrNull()
            }.maxByOrNull { it.time }
                ?.let { LocationCoordinates(it.latitude, it.longitude, it.accuracy) }
        }.getOrNull()
    }

    private fun getGpsLocation(): String {
        if (!hasPermission(Manifest.permission.ACCESS_FINE_LOCATION) && !hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)) {
            return "Location permission is required before I can read GPS hands-free."
        }
        return try {
            val location = readLastKnownLocation()
            if (location == null) {
                "No last known location is available yet."
            } else {
                "Current location: latitude=${location.latitude}, longitude=${location.longitude}, accuracy=${location.accuracyMeters}m."
            }
        } catch (e: Exception) {
            "Could not read location: ${e.message}"
        }
    }

    private fun getWeather(location: String): String {
        val query = if (location.isNotBlank()) {
            location
        } else {
            readLastKnownLocation()?.let { "${it.latitude},${it.longitude}" }.orEmpty()
        }
        if (query.isBlank()) return "I need a location or GPS permission to fetch weather."
        return fetchText("https://wttr.in/${URLEncoder.encode(query, "UTF-8")}?format=3", "weather")
    }

    private fun webSearch(query: String): String {
        val cleanQuery = query.trim()
        if (cleanQuery.isBlank()) return "I need a query to search the web."

        val instantAnswer = fetchDuckDuckGoInstantAnswer(cleanQuery)
        if (instantAnswer.isNotBlank()) {
            return instantAnswer.take(WEB_SEARCH_MAX_OUTPUT_CHARS)
        }

        val searchUrl = "https://duckduckgo.com/?q=${URLEncoder.encode(cleanQuery, "UTF-8")}"
        val result = launchIntent(Intent(Intent.ACTION_VIEW, Uri.parse(searchUrl)), "web search")
        return if (result.startsWith("Could not", ignoreCase = true)) {
            "I could not fetch an instant answer or open web search for \"$cleanQuery\": $result"
        } else {
            "Opened web search for \"$cleanQuery\"."
        }
    }

    private fun fetchDuckDuckGoInstantAnswer(query: String): String {
        val url = "https://api.duckduckgo.com/?q=${URLEncoder.encode(query, "UTF-8")}&format=json&no_html=1&skip_disambig=1"
        val raw = fetchText(url, "DuckDuckGo instant answer", maxChars = 120_000)
        if (raw.startsWith("Could not", ignoreCase = true)) return raw

        val parsed = runCatching { gson.fromJson(raw, Map::class.java) as Map<*, *> }.getOrNull() ?: return ""
        val heading = parsed["Heading"]?.toString().orEmpty()
        val abstract = parsed["AbstractText"]?.toString().orEmpty()
        val related = (parsed["RelatedTopics"] as? List<*>)?.firstOrNull()?.let { it as? Map<*, *> }?.get("Text")?.toString().orEmpty()
        return listOf(heading, abstract, related)
            .filter { it.isNotBlank() }
            .joinToString("\n")
            .take(1200)
    }

    private fun fetchText(url: String, label: String, maxChars: Int = Int.MAX_VALUE): String = try {
        val request = Request.Builder()
            .url(url)
            .get()
            .header("User-Agent", WEB_SEARCH_USER_AGENT)
            .header("Accept-Language", "de,en;q=0.8")
            .build()
        getWebFetchClient().newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                "Could not fetch $label: HTTP ${response.code}"
            } else {
                val text = response.body?.string()?.trim().orEmpty()
                if (maxChars == Int.MAX_VALUE) text else text.take(maxChars)
            }
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

    private fun createReminder(text: String, dueTime: String, dueAtMillis: Long?): String {
        if (text.isBlank()) return "Reminder text cannot be empty."
        val prefs = context.getSharedPreferences("assistant_reminders", Context.MODE_PRIVATE)
        val existing = prefs.getStringSet("items", emptySet()).orEmpty().toMutableSet()
        existing.add("${System.currentTimeMillis()}|$text|$dueTime|${dueAtMillis ?: ""}")
        prefs.edit().putStringSet("items", existing).apply()

        val savedMessage = "Reminder saved: $text${if (dueTime.isNotBlank()) " ($dueTime)" else ""}."
        val androidResult = scheduleAndroidReminder(text, dueAtMillis)
        return listOf(savedMessage, androidResult).filter { it.isNotBlank() }.joinToString(" ")
    }

    private fun scheduleAndroidReminder(text: String, dueAtMillis: Long?): String =
        dueAtMillis?.takeIf { it > System.currentTimeMillis() }?.let { triggerAtMillis ->
            scheduleReminderNotification(text, triggerAtMillis)
        } ?: ""

    private fun scheduleReminderNotification(text: String, triggerAtMillis: Long): String {
        return try {
            val alarmManager = context.getSystemService(AlarmManager::class.java)
                ?: return "Reminder saved, but Android AlarmManager is unavailable."
            val notificationId = (System.currentTimeMillis() and Int.MAX_VALUE.toLong()).toInt()
            val intent = Intent(context, ReminderReceiver::class.java)
                .putExtra(ReminderReceiver.EXTRA_TEXT, text)
                .putExtra(ReminderReceiver.EXTRA_NOTIFICATION_ID, notificationId)
            val pendingIntent = PendingIntent.getBroadcast(
                context,
                notificationId,
                intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )

            alarmManager.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, triggerAtMillis, pendingIntent)
            val formattedTime = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT, Locale.GERMANY)
                .format(Date(triggerAtMillis))
            "Reminder notification scheduled automatically for $formattedTime."
        } catch (e: Exception) {
            Log.e(appTag, "Could not schedule reminder notification", e)
            "Reminder saved, but I could not schedule the notification: ${e.message}"
        }
    }

    private fun listReminders(): String {
        val prefs = context.getSharedPreferences("assistant_reminders", Context.MODE_PRIVATE)
        val reminders = prefs.getStringSet("items", emptySet()).orEmpty().sorted().map { item ->
            val parts = item.split("|", limit = 4)
            val dueText = parts.getOrNull(2)?.takeIf { it.isNotBlank() }
            val dueMillis = parts.getOrNull(3)?.toLongOrNull()
            val dueLabel = dueText ?: dueMillis?.let {
                DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT, Locale.GERMANY).format(Date(it))
            }
            "- ${parts.getOrNull(1).orEmpty()}${dueLabel?.let { " ($it)" }.orEmpty()}"
        }
        return if (reminders.isEmpty()) "No reminders saved." else reminders.joinToString("\n")
    }

    private fun sendEmail(to: String, subject: String, body: String): String {
        if (to.isBlank()) return "Ich brauche eine E-Mail-Adresse."
        val mailUri = Uri.parse(
            "mailto:${Uri.encode(to)}" +
                "?subject=${Uri.encode(subject)}" +
                "&body=${Uri.encode(body)}"
        )
        val intents = listOf(
            Intent(Intent.ACTION_SENDTO, mailUri).setPackage("com.google.android.gm"),
            Intent(Intent.ACTION_SENDTO, mailUri),
            Intent(Intent.ACTION_SEND)
                .setType("message/rfc822")
                .putExtra(Intent.EXTRA_EMAIL, arrayOf(to))
                .putExtra(Intent.EXTRA_SUBJECT, subject)
                .putExtra(Intent.EXTRA_TEXT, body)
                .setPackage("com.google.android.gm"),
            Intent(Intent.ACTION_SEND)
                .setType("message/rfc822")
                .putExtra(Intent.EXTRA_EMAIL, arrayOf(to))
                .putExtra(Intent.EXTRA_SUBJECT, subject)
                .putExtra(Intent.EXTRA_TEXT, body)
        )

        intents.forEach { intent ->
            val result = launchIntent(intent, "E-Mail-Entwurf")
            if (!result.startsWith("Could not")) {
                return "Opened email draft for $to. Android/Gmail requires you to confirm Send; silent Gmail sending needs OAuth or SMTP configuration."
            }
        }
        return "Could not open an email app for $to."
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

    private fun createCalendarEvent(
        title: String,
        start: String,
        end: String,
        startEpochMillis: Long?,
        endEpochMillis: Long?,
        location: String,
        notes: String
    ): String {
        if (title.isBlank()) return "Ich brauche einen Termintitel."
        val intent = Intent(Intent.ACTION_INSERT)
            .setData(CalendarContract.Events.CONTENT_URI)
            .putExtra(CalendarContract.Events.TITLE, title)
            .putExtra(CalendarContract.Events.EVENT_LOCATION, location)
            .putExtra(CalendarContract.Events.DESCRIPTION, listOf(notes, start.takeIf { it.isNotBlank() }?.let { "Start: $it" }, end.takeIf { it.isNotBlank() }?.let { "Ende: $it" }).filterNotNull().joinToString("\n"))
        startEpochMillis?.takeIf { it > 0 }?.let { startMillis ->
            intent.putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, startMillis)
            val endMillis = endEpochMillis?.takeIf { it > startMillis } ?: (startMillis + 30 * 60 * 1000L)
            intent.putExtra(CalendarContract.EXTRA_EVENT_END_TIME, endMillis)
        }
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
        persistConversation: Boolean = false,
        conversationHistory: List<Message> = emptyList()
    ) {
        Log.d(appTag, "OpenAI Streaming API called")
        Log.d(appTag, "Instruction: $instruction")
        Log.d(appTag, "Has image: ${image != null}")

        Thread {
            var builtChatContext: ChatRequestContext? = null
            try {
                val chatContext = buildChatRequestContext(
                    instruction = instruction,
                    image = image,
                    allowGlassesPhotoTool = allowGlassesPhotoTool,
                    includePersistedHistory = includePersistedHistory,
                    persistConversation = persistConversation,
                    conversationHistory = conversationHistory
                )
                builtChatContext = chatContext
                if (chatContext.deferred) {
                    Log.d(appTag, "Assistant tool deferred this request; waiting for local follow-up")
                    return@Thread
                }
                chatContext.directResponse?.takeIf { it.isNotBlank() }?.let { directResponse ->
                    Log.d(appTag, "Using direct tool-planning response")
                    deliverChatResponse(
                        response = directResponse,
                        chatContext = chatContext,
                        instruction = instruction,
                        persistConversation = persistConversation,
                        emitStreamingChunk = true
                    )
                    return@Thread
                }

                // Convert request to JSON
                val jsonBody = gson.toJson(chatContext.request)
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
                        val retryResponse = recoverImageResponseWithoutStreaming(chatContext, "Streaming API call failed")
                        if (retryResponse.isNotBlank()) {
                            deliverChatResponse(
                                response = retryResponse,
                                chatContext = chatContext,
                                instruction = instruction,
                                persistConversation = persistConversation,
                                emitStreamingChunk = true
                            )
                            return@Thread
                        }
                        deliverFailureResponse(
                            buildAssistantFailureResponse(
                                reason = "Der KI-Dienst hat HTTP ${response.code} zurückgegeben",
                                detail = errorBody,
                                toolResults = chatContext.toolResults
                            ),
                            chatContext,
                            instruction,
                            persistConversation
                        )
                        return@Thread
                    }

                    // Read the streaming response
                    val source = response.body?.source()
                    if (source == null) {
                        Log.e(appTag, "Empty response body")
                        val retryResponse = recoverImageResponseWithoutStreaming(chatContext, "Streaming response body was empty")
                        if (retryResponse.isNotBlank()) {
                            deliverChatResponse(
                                response = retryResponse,
                                chatContext = chatContext,
                                instruction = instruction,
                                persistConversation = persistConversation,
                                emitStreamingChunk = true
                            )
                            return@Thread
                        }
                        deliverFailureResponse(
                            buildAssistantFailureResponse(
                                reason = "Der Antwortkörper war leer",
                                toolResults = chatContext.toolResults
                            ),
                            chatContext,
                            instruction,
                            persistConversation
                        )
                        return@Thread
                    }

                    val fullResponse = StringBuilder()

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
                        val retryResponse = recoverImageResponseWithoutStreaming(chatContext, "Streaming read timed out")
                        if (retryResponse.isNotBlank()) {
                            deliverChatResponse(
                                response = retryResponse,
                                chatContext = chatContext,
                                instruction = instruction,
                                persistConversation = persistConversation,
                                emitStreamingChunk = true
                            )
                            return@Thread
                        }
                        deliverFailureResponse(
                            buildAssistantFailureResponse(
                                reason = "Der Antwortstream ist abgelaufen",
                                detail = "Der Server hat vermutlich kein kompatibles SSE-Streaming gesendet.",
                                toolResults = chatContext.toolResults
                            ),
                            chatContext,
                            instruction,
                            persistConversation
                        )
                        return@Thread
                    }

                    // Also send the full response to the non-streaming callback for backwards compatibility
                    val finalResponse = fullResponse.toString().trim()
                    if (finalResponse.isNotBlank()) {
                        Log.d(appTag, "Full streaming response: $finalResponse")
                        deliverChatResponse(
                            response = finalResponse,
                            chatContext = chatContext,
                            instruction = instruction,
                            persistConversation = persistConversation,
                            emitStreamingChunk = false
                        )
                    } else {
                        val retryResponse = recoverImageResponseWithoutStreaming(chatContext, "Streaming completed without text")
                        if (retryResponse.isNotBlank()) {
                            deliverChatResponse(
                                response = retryResponse,
                                chatContext = chatContext,
                                instruction = instruction,
                                persistConversation = persistConversation,
                                emitStreamingChunk = true
                            )
                            return@Thread
                        }

                        val fallbackResponse = buildToolResultFallback(chatContext.toolResults)
                        if (fallbackResponse.isNotBlank()) {
                            Log.w(appTag, "Streaming completed empty after tool call; using fallback response: $fallbackResponse")
                            deliverChatResponse(
                                response = fallbackResponse,
                                chatContext = chatContext,
                                instruction = instruction,
                                persistConversation = persistConversation,
                                emitStreamingChunk = true
                            )
                        } else {
                            Log.w(appTag, "Streaming completed but no content was received")
                            val genericFallback = buildAssistantFailureResponse(
                                reason = "Der Antwortstream war leer",
                                detail = "Die Anfrage wurde gesendet, aber es kam kein Text zurück.",
                                toolResults = chatContext.toolResults
                            )
                            deliverFailureResponse(
                                genericFallback,
                                chatContext,
                                instruction,
                                persistConversation
                            )
                        }
                    }
                }
            } catch (e: IOException) {
                Log.e(appTag, "Network error: ${e.message}", e)
                deliverFailureResponse(
                    buildAssistantFailureResponse(
                        reason = "Netzwerkfehler",
                        detail = e.message,
                        toolResults = builtChatContext?.toolResults.orEmpty()
                    ),
                    builtChatContext,
                    instruction,
                    persistConversation
                )
            } catch (e: Exception) {
                Log.e(appTag, "Error calling OpenAI Streaming API: ${e.message}", e)
                deliverFailureResponse(
                    buildAssistantFailureResponse(
                        reason = "Interner Fehler bei der KI-Anfrage",
                        detail = e.message,
                        toolResults = builtChatContext?.toolResults.orEmpty()
                    ),
                    builtChatContext,
                    instruction,
                    persistConversation
                )
            }
        }.start()
    }

    private fun buildChatRequestContext(
        instruction: String,
        image: Bitmap?,
        allowGlassesPhotoTool: Boolean,
        includePersistedHistory: Boolean,
        persistConversation: Boolean,
        conversationHistory: List<Message>
    ): ChatRequestContext {
        val hasImage = image != null
        val contentList = mutableListOf<Content>()

        if (image != null) {
            val base64Image = bitmapToBase64(image)
            val imageDataUrl = "data:image/png;base64,$base64Image"
            contentList.add(
                Content(
                    type = "image_url",
                    imageUrl = ImageUrl(url = imageDataUrl)
                )
            )
            contentList.add(Content(type = "text", text = visionInstruction(instruction)))
        } else {
            contentList.add(Content(type = "text", text = instruction))
        }

        val requestChatId = activeChatIdOrNull()
        val messagesList = mutableListOf<Message>()
        val sessionPrompt = when {
            hasImage ->
                "Diese Anfrage enthält ein frisches Kamerabild. Verwende dieses Bild als primäre Quelle und ziehe keine älteren Bildbeschreibungen oder Tool-Ergebnisse heran."
            requestChatId != null ->
                "Du bist in einem explizit gewählten persistenten Chat. Verwende dessen Verlauf. Nutze Chat-Verwaltungstools nur, wenn der Nutzer neue Chats, Chatlisten, Umbenennungen, Löschungen, Wechsel oder das Verlassen des persistenten Chats wünscht."
            conversationHistory.isNotEmpty() ->
                "Du bist in der ephemeren Hauptkonversation. Du hast nur den Verlauf dieser gerade aktiven Brillen-Konversation. Erstelle oder betrete persistente Chats nur bei ausdrücklicher Nutzerbitte."
            else ->
                "Diese Anfrage ist Teil der ephemeren Hauptkonversation und hat keinen gespeicherten Chat-Verlauf. Erstelle oder betrete persistente Chats nur bei ausdrücklicher Nutzerbitte."
        }
        val systemPrompt = getSystemPrompt() +
            "\n\nDu heißt Neuro. Deutsch (Österreich), kurz und freihändig. ${currentAssistantTimeHint()} " +
            "Nutze Tools nur bei klarer Handlung, nie für kurze Tests wie hallo/test/ping. Sichtfragen brauchen snap_glasses_photo. " +
            "Navigation: start_navigation mit komplett gehörtem Ziel; public_transit ist Standard, foot nur bei Fußweg. Folgeauswahl wie eins/zwei/drei nutzt confirm_navigation_destination. " +
            "Öffne keine externe Navi-App. Kontaktname reicht für Anruf/SMS. Timer/Erinnerungen/Kalender: berechne exakte seconds oder epoch_millis selbst und übergib numerische Zeitwerte als Dezimalziffern-Strings. Erwarte keine Textinterpretation durch die App. " +
            "Aktuelles/Web/Preise/Öffnungszeiten: web_search. Persistente Chats: new_chat, list_chats, switch_chat, rename_chat, leave_chat und delete_chat nur bei ausdrücklichem Wunsch. $sessionPrompt"
        messagesList.add(Message(role = "system", content = systemPrompt))

        val includeHistoryForRequest = !hasImage
        if (includeHistoryForRequest) {
            if (requestChatId != null) {
                messagesList.addAll(chatHistoryMessages(requestChatId))
            } else if (conversationHistory.isNotEmpty()) {
                messagesList.addAll(conversationHistory.takeLast(20))
            }
        } else if (hasImage) {
            Log.d(appTag, "Skipping chat history for fresh image request")
        }

        messagesList.add(Message(role = "user", content = contentList))

        val toolResolution = if (!hasImage) {
            resolveToolCallsIfNeeded(
                messagesList,
                instruction,
                allowGlassesPhotoTool,
                includeChatTools = true
            )
        } else {
            val reason = if (hasImage) "image request" else "text request"
            Log.d(appTag, "Skipping tool planning for $reason: $instruction")
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

        val request = OpenAIRequest(
            model = getVlmModel(),
            messages = toolMessages,
            maxTokens = getVlmMaxTokens(),
            stream = true
        )

        return ChatRequestContext(
            request = request,
            toolResults = toolResolution.results,
            requestChatId = requestChatId,
            hasImage = hasImage,
            directResponse = toolResolution.directResponse,
            deferred = toolResolution.deferred
        )
    }

    private fun currentAssistantTimeHint(): String {
        val now = System.currentTimeMillis()
        val formatted = DateFormat.getDateTimeInstance(DateFormat.FULL, DateFormat.LONG, Locale.GERMANY)
            .format(Date(now))
        return "Aktuelle lokale Zeit: $formatted; epoch_ms=$now."
    }

    private fun recoverImageResponseWithoutStreaming(chatContext: ChatRequestContext, reason: String): String {
        if (!chatContext.hasImage) return ""
        Log.w(appTag, "$reason; retrying image request without streaming")
        return try {
            callOpenAINonStreaming(chatContext.request)
        } catch (e: Exception) {
            Log.e(appTag, "Non-streaming image retry failed", e)
            ""
        }
    }

    private fun callOpenAINonStreaming(request: OpenAIRequest): String {
        val nonStreamingRequest = request.copy(stream = false, tools = null, toolChoice = null)
        val jsonBody = gson.toJson(nonStreamingRequest)
        Log.d(appTag, "Non-streaming retry request JSON: $jsonBody")

        val requestBody = jsonBody.toRequestBody("application/json".toMediaType())
        val httpRequest = Request.Builder()
            .url(getChatApiUrl())
            .addHeader("Content-Type", "application/json")
            .addHeader("Authorization", "Bearer ${getApiToken()}")
            .post(requestBody)
            .build()

        getClient().newCall(httpRequest).execute().use { response ->
            if (!response.isSuccessful) {
                val errorBody = response.body?.string() ?: "Unknown error"
                Log.e(appTag, "Non-streaming retry failed: ${response.code} - $errorBody")
                return ""
            }

            val responseBody = response.body?.string().orEmpty()
            if (responseBody.isBlank()) {
                Log.e(appTag, "Non-streaming retry returned an empty body")
                return ""
            }

            val chatResponse = gson.fromJson(responseBody, ChatCompletionResponse::class.java)
            val message = chatResponse.choices.firstOrNull()?.message
            return messageContentToText(message?.content).trim()
        }
    }

    private fun messageContentToText(content: Any?): String = when (content) {
        null -> ""
        is String -> content
        is Content -> content.text.orEmpty()
        is List<*> -> content.joinToString("") { messageContentToText(it) }
        is Map<*, *> -> {
            val text = content["text"]?.toString().orEmpty()
            if (text.isNotBlank()) text else content["refusal"]?.toString().orEmpty()
        }
        else -> content.toString()
    }

    private fun deliverChatResponse(
        response: String,
        chatContext: ChatRequestContext,
        instruction: String,
        persistConversation: Boolean,
        emitStreamingChunk: Boolean
    ) {
        val finalResponse = response.trim()
        if (finalResponse.isBlank()) return

        if (emitStreamingChunk) {
            listener?.onOpenAIStreamingChunk(finalResponse, true)
        }
        val persistentModeEnded = chatContext.toolResults.any {
            it.startsWith("leave_chat:") || it.startsWith("delete_chat:")
        }
        val targetChatId = if (persistentModeEnded) null else activeChatIdOrNull() ?: chatContext.requestChatId
        if (targetChatId != null) {
            appendChatMessages(
                targetChatId,
                listOf(
                    PersistedChatMessage("user", instruction),
                    PersistedChatMessage("assistant", finalResponse)
                )
            )
        }
        listener?.onOpenAIResponse(finalResponse)
    }

    private fun deliverFailureResponse(
        response: String,
        chatContext: ChatRequestContext?,
        instruction: String,
        persistConversation: Boolean
    ) {
        val finalResponse = response.trim()
        if (finalResponse.isBlank()) return

        if (chatContext != null) {
            deliverChatResponse(
                response = finalResponse,
                chatContext = chatContext,
                instruction = instruction,
                persistConversation = persistConversation,
                emitStreamingChunk = true
            )
            return
        }

        listener?.onOpenAIStreamingChunk(finalResponse, true)
        listener?.onOpenAIResponse(finalResponse)
    }


    private fun resolveToolCallsIfNeeded(
        messagesList: MutableList<Message>,
        instruction: String,
        allowGlassesPhotoTool: Boolean,
        includeChatTools: Boolean
    ): ToolResolutionResult {
        val tools = assistantTools(allowGlassesPhotoTool, includeChatTools)
        val toolResults = mutableListOf<String>()

        return try {
            repeat(MAX_TOOL_ROUNDS) { round ->
                val request = OpenAIRequest(
                    model = getVlmModel(),
                    messages = messagesList,
                    maxTokens = TOOL_PLANNING_MAX_TOKENS,
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

                getToolPlanningClient().newCall(httpRequest).execute().use { response ->
                    if (!response.isSuccessful) {
                        Log.w(appTag, "Tool planning skipped: ${response.code} - ${response.body?.string()}")
                        return ToolResolutionResult(messagesList, toolResults)
                    }

                    val responseBody = response.body?.string() ?: return ToolResolutionResult(messagesList, toolResults)
                    val chatResponse = gson.fromJson(responseBody, ChatCompletionResponse::class.java)
                    val assistantMessage = chatResponse.choices.firstOrNull()?.message ?: return ToolResolutionResult(messagesList, toolResults)
                    val toolCalls = assistantMessage.toolCalls.orEmpty()

                    if (toolCalls.isEmpty()) {
                        val directText = messageContentToText(assistantMessage.content).trim()
                        if (directText.isNotBlank()) {
                            Log.d(appTag, "Tool planning produced direct text without tool calls")
                            return ToolResolutionResult(
                                messages = messagesList,
                                results = toolResults,
                                directResponse = directText
                            )
                        }
                        return ToolResolutionResult(messagesList, toolResults)
                    }

                    messagesList.add(assistantMessage)

                    Log.d(appTag, "Executing ${toolCalls.size} tool call(s), round ${round + 1}")
                    toolCalls.forEach { toolCall ->
                        val result = executeAssistantTool(toolCall, instruction)
                        if (result == DEFERRED_PHOTO_TOOL_RESULT || result == LOCAL_TOOL_HANDLED_RESULT) {
                            Log.d(appTag, "Tool ${toolCall.function.name} handled locally; deferring final assistant response")
                            return ToolResolutionResult(messagesList, toolResults, deferred = true)
                        }
                        toolResults += "${toolCall.function.name}: $result"
                        listener?.onAssistantToolResult(toolCall.function.name, result)
                        if (shouldDeferAfterToolResult(toolCall.function.name, result)) {
                            Log.d(appTag, "Tool ${toolCall.function.name} completed locally; deferring final assistant response")
                            return ToolResolutionResult(messagesList, toolResults, deferred = true)
                        }
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
            var call: Call? = null
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
        val normalized = text.trim()
        if (normalized.length <= MAX_TTS_INPUT_CHARS) return normalized
        return normalized.take(MAX_TTS_INPUT_CHARS).trim()
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
