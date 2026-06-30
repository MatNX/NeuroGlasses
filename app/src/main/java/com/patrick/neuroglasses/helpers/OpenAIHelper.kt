package com.patrick.neuroglasses.helpers

import android.annotation.SuppressLint
import android.Manifest
import android.app.Activity
import android.app.ActivityOptions
import android.app.AlarmManager
import android.content.ContentUris
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.app.KeyguardManager
import android.app.PendingIntent
import android.net.Uri
import android.os.Build
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
import com.google.gson.JsonElement
import com.google.gson.annotations.SerializedName
import com.patrick.neuroglasses.R
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
import java.time.DayOfWeek
import java.time.Instant
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.LocalTime
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.text.DateFormat
import java.text.Normalizer
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import java.util.UUID
import kotlin.math.cos
import org.json.JSONArray
import org.json.JSONObject

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

data class SemanticMemoryRecord(
    val id: String,
    val content: String,
    val category: String,
    val tags: List<String> = emptyList(),
    val importance: Int = 3,
    val createdAt: Long,
    val updatedAt: Long
)

data class SavedPlaceRecord(
    val id: String,
    val name: String,
    val description: String = "",
    val address: String = "",
    val latitude: Double,
    val longitude: Double,
    val createdAt: Long,
    val updatedAt: Long
)

/**
 * Groq AI Helper
 * Handles AI-related API calls including ASR (speech-to-text) and OpenAI chat completion
 */
class OpenAIHelper(private val context: Context, private val appTag: String = "OpenAIHelper") {

    data class DebugToolSpec(
        val name: String,
        val internalName: String,
        val description: String,
        val argumentTemplate: String
    )

    data class DebugToolResult(
        val requestedName: String,
        val exposedName: String,
        val internalName: String,
        val arguments: Map<String, String>,
        val result: String,
        val durationMillis: Long,
        val error: String? = null
    )

    private val gson = Gson()
    private val chatPrefs by lazy { context.getSharedPreferences("assistant_chats", Context.MODE_PRIVATE) }
    private val memoryPrefs by lazy { context.getSharedPreferences("assistant_semantic_memory", Context.MODE_PRIVATE) }
    private val placePrefs by lazy { context.getSharedPreferences("assistant_saved_places", Context.MODE_PRIVATE) }
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
        private const val TOOL_LAUNCH_CHANNEL_ID = "neuro_tool_launch"
        private const val TOOL_LAUNCH_NOTIFICATION_ID = 44
        private const val TOOL_LAUNCH_NOTIFICATION_TIMEOUT_MS = 15_000L
        private const val PREF_ACTIVE_CHAT_ID = "active_chat_id"
        private const val PREF_ACTIVE_CHAT_EXPLICIT = "active_chat_explicit"
        private const val WEB_SEARCH_MAX_OUTPUT_CHARS = 5000
        private const val WEB_SEARCH_RESULT_LIMIT = 5
        private const val MEMORY_CONTEXT_LIMIT = 5
        private const val MEMORY_TOOL_RESULT_LIMIT = 8
        private const val SAVED_PLACE_RESULT_LIMIT = 8
        private const val NEARBY_RESULT_LIMIT = 6
        private const val TRANSIT_RESULT_LIMIT = 8
        private const val OEBB_API_BASE_URL = "https://v6.oebb.transport.rest"
        private const val OEBB_SCOTTY_BASE_URL = "https://fahrplan.oebb.at"
        private const val OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
        private const val OVERPASS_KUMI_API_URL = "https://overpass.kumi.systems/api/interpreter"
        private const val DEFAULT_CALENDAR_EVENT_DURATION_MILLIS = 30 * 60 * 1000L
        private const val WEB_SEARCH_USER_AGENT =
            "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) NeuroGlasses/1.0 Mobile Safari/537.36"
        private val SEMANTIC_STOP_WORDS = setOf(
            "der", "die", "das", "und", "oder", "aber", "ich", "du", "wir", "mir", "mich",
            "mein", "meine", "dein", "eine", "einen", "einem", "ist", "sind", "war", "mit",
            "wie", "wer", "was",
            "for", "the", "and", "or", "but", "this", "that", "what", "where", "when", "how",
            "about", "into", "from", "have", "has", "had", "please", "bitte"
        )
        private val OVERPASS_API_URLS = listOf(OVERPASS_API_URL, OVERPASS_KUMI_API_URL)
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

    private data class WebSearchResult(
        val title: String,
        val url: String,
        val snippet: String
    )

    private data class NearbyPlace(
        val name: String,
        val category: String,
        val latitude: Double,
        val longitude: Double,
        val distanceMeters: Float,
        val detail: String
    )

    private data class WritableCalendarCandidate(
        val id: Long,
        val displayName: String,
        val accessLevel: Int,
        val isPrimary: Boolean,
        val isOwner: Boolean
    )

    private data class TextFetchResult(
        val text: String? = null,
        val httpCode: Int? = null,
        val errorMessage: String? = null
    )

    private data class TransitStop(
        val id: String,
        val name: String,
        val distanceMeters: Double? = null
    )

    private data class TransitStopLookup(
        val stops: List<TransitStop> = emptyList(),
        val error: String? = null
    )

    private data class TransitDeparturesLookup(
        val departures: List<String> = emptyList(),
        val error: String? = null
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
        } ?: return "Chat nicht gefunden. Nutze chats_auflisten, um verfügbare Chat-IDs und Titel zu sehen."
        chatPrefs.edit()
            .putString(PREF_ACTIVE_CHAT_ID, session.id)
            .putBoolean(PREF_ACTIVE_CHAT_EXPLICIT, true)
            .apply()
        return "Zu Chat '${session.title}' (${session.id}) gewechselt."
    }

    private fun listChats(): String {
        val active = activeChatIdOrNull()
        val sessions = getChatSessions()
        if (sessions.isEmpty()) return "Es gibt noch keine Chats."
        return sessions.joinToString("\n") { session ->
            val marker = if (session.id == active) "*" else "-"
            "$marker ${session.title} [${session.id}] (${session.messages.size} Nachrichten)"
        }
    }

    private fun renameChat(chatId: String?, title: String): String {
        if (title.isBlank()) return "Ein neuer Chattitel ist erforderlich."
        val targetId = chatId?.takeIf { it.isNotBlank() } ?: activeChatIdOrNull()
            ?: return "Kein persistenter Chat ist aktiv."
        val updated = updateChatSession(targetId) { it.copy(title = title) }
        return if (updated == null) "Chat nicht gefunden." else "Chat in '${updated.title}' umbenannt."
    }

    private fun leaveChat(): String {
        leavePersistentConversation()
        return "Persistenten Chat verlassen. Die Hauptkonversation ist wieder flüchtig."
    }

    private fun deleteChat(chatId: String?): String {
        val targetId = chatId?.takeIf { it.isNotBlank() } ?: activeChatIdOrNull()
            ?: return "Kein persistenter Chat ist aktiv."
        return if (deletePersistentConversation(targetId)) {
            "Persistenten Chat gelöscht."
        } else {
            "Chat nicht gefunden."
        }
    }

    private fun getSemanticMemories(): MutableList<SemanticMemoryRecord> {
        val json = memoryPrefs.getString("items", null) ?: return mutableListOf()
        return runCatching {
            gson.fromJson(json, Array<SemanticMemoryRecord>::class.java).toMutableList()
        }.getOrElse {
            Log.w(appTag, "Could not read semantic memories", it)
            mutableListOf()
        }
    }

    private fun saveSemanticMemories(memories: List<SemanticMemoryRecord>) {
        memoryPrefs.edit().putString("items", gson.toJson(memories)).apply()
    }

    private fun rememberMemory(content: String, category: String, tagsRaw: String, importanceRaw: String): String {
        val cleanContent = content.trim()
        if (cleanContent.isBlank()) return "Der Erinnerungsinhalt darf nicht leer sein."

        val now = System.currentTimeMillis()
        val cleanCategory = category.trim().ifBlank { inferMemoryCategory(cleanContent) }.take(40)
        val tags = enrichedMemoryTags(parseStringList(tagsRaw), cleanContent, cleanCategory).take(12)
        val requestedImportance = importanceRaw.toIntOrNull()?.coerceIn(1, 5) ?: 3
        val importance = maxOf(requestedImportance, inferredMemoryImportance(cleanContent, cleanCategory))
        val memories = getSemanticMemories()

        val existingIndex = memories.indexOfFirst { memory ->
            semanticText(memory.content) == semanticText(cleanContent) ||
                semanticOverlapScore(cleanContent, memory.content, memory.tags, memory.category) >= 18.0
        }
        val record = if (existingIndex >= 0) {
            memories[existingIndex].copy(
                content = cleanContent,
                category = cleanCategory,
                tags = tags.ifEmpty { memories[existingIndex].tags },
                importance = maxOf(importance, memories[existingIndex].importance),
                updatedAt = now
            )
        } else {
            SemanticMemoryRecord(
                id = shortId(),
                content = cleanContent,
                category = cleanCategory,
                tags = tags,
                importance = importance,
                createdAt = now,
                updatedAt = now
            )
        }

        if (existingIndex >= 0) {
            memories[existingIndex] = record
        } else {
            memories.add(0, record)
        }
        saveSemanticMemories(memories.sortedWith(compareByDescending<SemanticMemoryRecord> { it.importance }.thenByDescending { it.updatedAt }))
        return "Erinnerung gespeichert [${record.id}]: ${record.content}"
    }

    private fun searchMemory(query: String, category: String, limitRaw: String): String {
        val memories = rankedMemories(query, category, limitRaw.toIntOrNull() ?: MEMORY_TOOL_RESULT_LIMIT)
        if (memories.isEmpty()) return "Keine passenden Erinnerungen gefunden."
        return memories.joinToString("\n") { memory ->
            "- [${memory.id}] ${memory.content}${memory.categoryLabel()}${memory.tagsLabel()}"
        }
    }

    private fun listMemories(category: String, limitRaw: String): String {
        val cleanCategory = category.trim()
        val limit = (limitRaw.toIntOrNull() ?: MEMORY_TOOL_RESULT_LIMIT).coerceIn(1, 30)
        val memories = getSemanticMemories()
            .asSequence()
            .filter { cleanCategory.isBlank() || it.category.equals(cleanCategory, ignoreCase = true) }
            .sortedWith(compareByDescending<SemanticMemoryRecord> { it.importance }.thenByDescending { it.updatedAt })
            .take(limit)
            .toList()
        if (memories.isEmpty()) return "Keine Erinnerungen gespeichert."
        return memories.joinToString("\n") { memory ->
            "- [${memory.id}] ${memory.content}${memory.categoryLabel()}${memory.tagsLabel()}"
        }
    }

    private fun forgetMemory(selection: String): String {
        val cleanSelection = selection.trim()
        if (cleanSelection.isBlank()) return "Sag mir, welche Erinnerungs-ID oder welches Thema ich vergessen soll."
        val memories = getSemanticMemories()
        val index = memories.indexOfFirst { it.id.equals(cleanSelection, ignoreCase = true) }
            .takeIf { it >= 0 }
            ?: memories.withIndex()
                .maxByOrNull { (_, memory) ->
                    semanticOverlapScore(cleanSelection, memory.content, memory.tags, memory.category)
                }
                ?.takeIf { (_, memory) ->
                    semanticOverlapScore(cleanSelection, memory.content, memory.tags, memory.category) >= 3.0
                }
                ?.index
            ?: return "Keine passende Erinnerung gefunden."

        val removed = memories.removeAt(index)
        saveSemanticMemories(memories)
        return "Erinnerung vergessen [${removed.id}]: ${removed.content}"
    }

    private fun semanticMemoryContext(instruction: String): String {
        val memoryQuery = passiveMemoryQuery(instruction)
        val relevant = rankedMemories(memoryQuery, "", MEMORY_CONTEXT_LIMIT)
            .filter { semanticOverlapScore(memoryQuery, it.content, it.tags, it.category) >= 2.0 || it.importance >= 5 }
            .ifEmpty {
                if (looksLikeIdentityMemoryQuestion(instruction)) {
                    getSemanticMemories()
                        .filter { isIdentityMemory(it.content, it.category, it.tags) }
                        .sortedWith(compareByDescending<SemanticMemoryRecord> { it.importance }.thenByDescending { it.updatedAt })
                        .take(MEMORY_CONTEXT_LIMIT)
                } else {
                    emptyList()
                }
            }
        if (relevant.isEmpty()) return ""
        return relevant.joinToString(
            prefix = "Relevante gespeicherte Erinnerungen:\n",
            separator = "\n"
        ) { memory ->
            "- ${memory.content}${memory.categoryLabel()}${memory.tagsLabel()}"
        }
    }

    private fun rankedMemories(query: String, category: String, limit: Int): List<SemanticMemoryRecord> {
        val cleanCategory = category.trim()
        val boundedLimit = limit.coerceIn(1, 30)
        val memories = getSemanticMemories()
            .filter { cleanCategory.isBlank() || it.category.equals(cleanCategory, ignoreCase = true) }
        if (query.isBlank()) {
            return memories
                .sortedWith(compareByDescending<SemanticMemoryRecord> { it.importance }.thenByDescending { it.updatedAt })
                .take(boundedLimit)
        }
        return memories
            .map { memory -> memory to semanticOverlapScore(query, memory.content, memory.tags, memory.category) }
            .filter { (_, score) -> score > 0.0 }
            .sortedWith(compareByDescending<Pair<SemanticMemoryRecord, Double>> { it.second }
                .thenByDescending { it.first.importance }
                .thenByDescending { it.first.updatedAt })
            .take(boundedLimit)
            .map { it.first }
    }

    private fun SemanticMemoryRecord.categoryLabel(): String =
        category.takeIf { it.isNotBlank() && it != "general" }?.let { " ($it)" }.orEmpty()

    private fun SemanticMemoryRecord.tagsLabel(): String =
        tags.takeIf { it.isNotEmpty() }?.joinToString(prefix = " #", separator = " #").orEmpty()

    private fun getSavedPlaces(): MutableList<SavedPlaceRecord> {
        val json = placePrefs.getString("items", null) ?: return mutableListOf()
        return runCatching {
            gson.fromJson(json, Array<SavedPlaceRecord>::class.java).toMutableList()
        }.getOrElse {
            Log.w(appTag, "Could not read saved places", it)
            mutableListOf()
        }
    }

    private fun saveSavedPlaces(places: List<SavedPlaceRecord>) {
        placePrefs.edit().putString("items", gson.toJson(places)).apply()
    }

    private fun savePlace(
        name: String,
        description: String,
        address: String,
        latitudeRaw: String,
        longitudeRaw: String
    ): String {
        val coordinates = explicitCoordinates(latitudeRaw, longitudeRaw)
            ?: resolvedPlaceCoordinates(address)
            ?: readLastKnownLocation()?.let { LocationCoordinates(it.latitude, it.longitude, it.accuracyMeters) }
            ?: return "Ich brauche GPS-Berechtigung, Koordinaten oder eine auflösbare Adresse, bevor ich diesen Ort speichern kann."

        val now = System.currentTimeMillis()
        val cleanName = name.trim().ifBlank { defaultSavedPlaceName(description, address, coordinates, now) }
        val places = getSavedPlaces()
        val existingIndex = places.indexOfFirst { it.name.equals(cleanName, ignoreCase = true) }
        val record = SavedPlaceRecord(
            id = if (existingIndex >= 0) places[existingIndex].id else shortId(),
            name = cleanName,
            description = description.trim(),
            address = address.trim(),
            latitude = coordinates.latitude,
            longitude = coordinates.longitude,
            createdAt = if (existingIndex >= 0) places[existingIndex].createdAt else now,
            updatedAt = now
        )
        if (existingIndex >= 0) {
            places[existingIndex] = record
        } else {
            places.add(0, record)
        }
        saveSavedPlaces(places.sortedByDescending { it.updatedAt })
        return "Ort gespeichert [${record.id}]: ${record.name} bei ${formatCoordinates(record.latitude, record.longitude)}."
    }

    private fun listSavedPlaces(query: String, limitRaw: String): String {
        val limit = (limitRaw.toIntOrNull() ?: SAVED_PLACE_RESULT_LIMIT).coerceIn(1, 30)
        val places = rankedSavedPlaces(query, limit)
        if (places.isEmpty()) return "Keine gespeicherten Orte gefunden."
        return places.joinToString("\n") { place ->
            "- [${place.id}] ${place.name}: ${formatCoordinates(place.latitude, place.longitude)}${placeDetail(place)}"
        }
    }

    private fun navigateSavedPlace(nameOrId: String, mode: String): String {
        val place = findSavedPlace(nameOrId) ?: return "Gespeicherter Ort nicht gefunden."
        return startNavigationToSavedPlace(place, mode)
    }

    private fun startNavigationToSavedPlace(place: SavedPlaceRecord, mode: String): String {
        return RokidConnectionService.startNavigation(
            context = context,
            destination = place.name,
            mode = normalizeNavigationMode(mode),
            latitude = place.latitude,
            longitude = place.longitude
        )
    }

    private fun forgetSavedPlace(nameOrId: String): String {
        val clean = nameOrId.trim()
        if (clean.isBlank()) return "Sag mir, welchen gespeicherten Ort ich vergessen soll."
        val places = getSavedPlaces()
        val index = places.indexOfFirst {
            it.id.equals(clean, ignoreCase = true) || it.name.equals(clean, ignoreCase = true)
        }.takeIf { it >= 0 }
            ?: places.withIndex()
                .maxByOrNull { (_, place) ->
                    semanticOverlapScore(clean, "${place.name} ${place.description} ${place.address}", emptyList(), "place")
                }
                ?.takeIf { (_, place) ->
                    semanticOverlapScore(clean, "${place.name} ${place.description} ${place.address}", emptyList(), "place") >= 3.0
                }
                ?.index
            ?: return "Gespeicherter Ort nicht gefunden."

        val removed = places.removeAt(index)
        saveSavedPlaces(places)
        return "Gespeicherten Ort gelöscht [${removed.id}]: ${removed.name}."
    }

    private fun findSavedPlace(nameOrId: String): SavedPlaceRecord? =
        rankedSavedPlaces(nameOrId, 1).firstOrNull()

    private fun findSavedPlaceForNavigation(destination: String): SavedPlaceRecord? {
        val clean = destination.trim()
        if (clean.isBlank()) return null
        val normalized = semanticText(clean)
        val queryTokens = semanticTokens(clean)
        val places = getSavedPlaces()
        places.firstOrNull { place ->
            val nameTokens = semanticTokens(place.name)
            place.id.equals(clean, ignoreCase = true) ||
                semanticText(place.name) == normalized ||
                (nameTokens.isNotEmpty() && nameTokens.all { it in queryTokens })
        }?.let { return it }

        return rankedSavedPlaces(clean, 1)
            .firstOrNull()
            ?.takeIf { place ->
                semanticOverlapScore(clean, "${place.name} ${place.description} ${place.address}", emptyList(), "place") >= 6.0
            }
    }

    private fun rankedSavedPlaces(query: String, limit: Int): List<SavedPlaceRecord> {
        val places = getSavedPlaces()
        if (query.isBlank()) return places.sortedByDescending { it.updatedAt }.take(limit)
        return places
            .map { place ->
                place to semanticOverlapScore(
                    query,
                    "${place.name} ${place.description} ${place.address}",
                    emptyList(),
                    "place"
                )
            }
            .filter { (_, score) -> score > 0.0 }
            .sortedWith(compareByDescending<Pair<SavedPlaceRecord, Double>> { it.second }.thenByDescending { it.first.updatedAt })
            .take(limit)
            .map { it.first }
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

    fun availableDebugTools(): List<DebugToolSpec> =
        assistantTools(allowGlassesPhotoTool = false, includeChatTools = true)
            .map { tool ->
                val internalName = internalToolName(tool.function.name)
                DebugToolSpec(
                    name = tool.function.name,
                    internalName = internalName,
                    description = tool.function.description,
                    argumentTemplate = debugArgumentTemplate(tool.function.parameters)
                )
            }

    fun executeDebugTool(
        toolName: String,
        argumentsJson: String,
        instruction: String = ""
    ): DebugToolResult {
        val requestedName = toolName.trim()
        val internalName = internalToolName(requestedName)
        val exposedName = exposedToolName(internalName)
        val startedAt = System.currentTimeMillis()
        var decodedArguments = emptyMap<String, String>()

        return try {
            val normalizedArguments = normalizeDebugArguments(argumentsJson)
            decodedArguments = decodeToolArguments(normalizedArguments)
            val toolCall = ToolCall(
                id = "debug-${System.currentTimeMillis()}",
                type = "function",
                function = ToolCallFunction(
                    name = requestedName,
                    arguments = normalizedArguments
                )
            )
            val result = executeAssistantTool(
                toolCall = toolCall,
                instruction = instruction,
                notifyListener = false
            )
            DebugToolResult(
                requestedName = requestedName,
                exposedName = exposedName,
                internalName = internalName,
                arguments = decodedArguments,
                result = result,
                durationMillis = System.currentTimeMillis() - startedAt
            )
        } catch (e: Exception) {
            Log.e(appTag, "Debug tool call failed for $requestedName", e)
            DebugToolResult(
                requestedName = requestedName,
                exposedName = exposedName,
                internalName = internalName,
                arguments = decodedArguments,
                result = "Debug tool call failed: ${e.message}",
                durationMillis = System.currentTimeMillis() - startedAt,
                error = e.message
            )
        }
    }

    private fun normalizeDebugArguments(argumentsJson: String): String {
        val cleanArguments = argumentsJson.trim().ifBlank { "{}" }
        val decoded = gson.fromJson(cleanArguments, JsonElement::class.java)
        require(decoded != null && decoded.isJsonObject) { "Arguments must be a JSON object." }
        return decoded.asJsonObject.toString()
    }

    private fun debugArgumentTemplate(parameters: Map<String, Any>): String {
        val properties = parameters["properties"] as? Map<*, *> ?: return "{}"
        if (properties.isEmpty()) return "{}"

        val template = JSONObject()
        properties.forEach { (rawName, rawProperty) ->
            val name = rawName as? String ?: return@forEach
            val property = rawProperty as? Map<*, *>
            template.put(name, debugArgumentDefault(property))
        }
        return template.toString(2)
    }

    private fun debugArgumentDefault(property: Map<*, *>?): String {
        val enumValues = property?.get("enum") as? List<*>
        return enumValues?.firstOrNull()?.toString().orEmpty()
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

    private fun exposedToolName(internalName: String): String = when (internalName) {
        "place_phone_call" -> "telefon_anrufen"
        "send_sms" -> "sms_senden"
        "find_contact_phone" -> "kontakt_telefon_suchen"
        "stop_conversation" -> "konversation_beenden"
        "open_rokid_native_app" -> "rokid_app_oeffnen"
        "navigation" -> "navigation"
        "start_navigation" -> "navigation_starten"
        "confirm_navigation_destination" -> "navigationsziel_bestaetigen"
        "stop_navigation" -> "navigation_stoppen"
        "open_rokid_translator" -> "rokid_uebersetzer_oeffnen"
        "get_gps_location" -> "standort_abrufen"
        "get_weather" -> "wetter_abrufen"
        "web_search" -> "web_suche"
        "set_timer" -> "timer_stellen"
        "create_reminder" -> "erinnerung_erstellen"
        "list_reminders" -> "erinnerungen_auflisten"
        "create_calendar_event" -> "kalendertermin_erstellen"
        "open_app" -> "app_oeffnen"
        "share_text" -> "text_teilen"
        "get_battery_status" -> "akku_status"
        "memory" -> "gedaechtnis"
        "remember_memory" -> "merken"
        "search_memory" -> "gedaechtnis_suchen"
        "list_memories" -> "gedaechtnis_auflisten"
        "forget_memory" -> "vergessen"
        "saved_place" -> "orte"
        "save_place" -> "ort_speichern"
        "list_saved_places" -> "orte_auflisten"
        "navigate_saved_place" -> "zu_gespeichertem_ort_navigieren"
        "forget_saved_place" -> "ort_vergessen"
        "get_agenda" -> "agenda_abrufen"
        "find_nearby_places" -> "nahe_orte_finden"
        "get_transit_departures" -> "abfahrten_abrufen"
        "open_accessibility_settings" -> "bedienungshilfen_oeffnen"
        "snap_glasses_photo" -> "brillenfoto_aufnehmen"
        "chat" -> "chat"
        "new_chat" -> "chat_neu"
        "list_chats" -> "chats_auflisten"
        "switch_chat" -> "chat_wechseln"
        "rename_chat" -> "chat_umbenennen"
        "leave_chat" -> "chat_verlassen"
        "delete_chat" -> "chat_loeschen"
        else -> internalName
    }

    private fun internalToolName(toolName: String): String = when (toolName) {
        "telefon_anrufen" -> "place_phone_call"
        "sms_senden" -> "send_sms"
        "kontakt_telefon_suchen" -> "find_contact_phone"
        "konversation_beenden" -> "stop_conversation"
        "rokid_app_oeffnen" -> "open_rokid_native_app"
        "navigation" -> "navigation"
        "navigation_starten" -> "start_navigation"
        "navigationsziel_bestaetigen" -> "confirm_navigation_destination"
        "navigation_stoppen" -> "stop_navigation"
        "rokid_uebersetzer_oeffnen" -> "open_rokid_translator"
        "standort_abrufen" -> "get_gps_location"
        "wetter_abrufen" -> "get_weather"
        "web_suche" -> "web_search"
        "timer_stellen" -> "set_timer"
        "erinnerung_erstellen" -> "create_reminder"
        "erinnerungen_auflisten" -> "list_reminders"
        "kalendertermin_erstellen" -> "create_calendar_event"
        "app_oeffnen" -> "open_app"
        "text_teilen" -> "share_text"
        "akku_status" -> "get_battery_status"
        "gedaechtnis", "memory" -> "memory"
        "merken" -> "remember_memory"
        "gedaechtnis_suchen" -> "search_memory"
        "gedaechtnis_auflisten" -> "list_memories"
        "vergessen" -> "forget_memory"
        "orte", "saved_places", "places" -> "saved_place"
        "ort_speichern" -> "save_place"
        "orte_auflisten" -> "list_saved_places"
        "zu_gespeichertem_ort_navigieren" -> "navigate_saved_place"
        "ort_vergessen" -> "forget_saved_place"
        "agenda_abrufen" -> "get_agenda"
        "nahe_orte_finden" -> "find_nearby_places"
        "abfahrten_abrufen" -> "get_transit_departures"
        "bedienungshilfen_oeffnen" -> "open_accessibility_settings"
        "brillenfoto_aufnehmen" -> "snap_glasses_photo"
        "chat" -> "chat"
        "chat_neu" -> "new_chat"
        "chats_auflisten" -> "list_chats"
        "chat_wechseln" -> "switch_chat"
        "chat_umbenennen" -> "rename_chat"
        "chat_verlassen" -> "leave_chat"
        "chat_loeschen" -> "delete_chat"
        else -> toolName
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
                name = exposedToolName("place_phone_call"),
                description = "Freihändig telefonieren. Ein Kontaktname reicht; frage nicht nach der Telefonnummer, wenn der Nutzer einen Namen gesagt hat. Die App löst Kontakte auf und fällt bei fehlender CALL_PHONE-Berechtigung auf den Dialer zurück.",
                parameters = objectSchema(listOf("recipient"), mapOf("recipient" to stringProp("Telefonnummer oder Kontaktname; Namen sind gültige Empfänger.")))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("send_sms"),
                description = "Freihändig eine SMS senden. Ein Kontaktname reicht; frage nicht nach der Telefonnummer, wenn der Nutzer einen Namen gesagt hat. Die App löst Kontakte auf und öffnet bei Bedarf den SMS-Composer.",
                parameters = objectSchema(listOf("recipient", "message"), mapOf(
                    "recipient" to stringProp("Telefonnummer oder Kontaktname; Namen sind gültige Empfänger."),
                    "message" to stringProp("Zu sendender Nachrichtentext.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("find_contact_phone"),
                description = "Kontakt im Telefonbuch suchen und die beste passende Telefonnummer zurückgeben.",
                parameters = objectSchema(listOf("recipient"), mapOf("recipient" to stringProp("Kontaktname, nach dem gesucht werden soll.")))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("stop_conversation"),
                description = "Aktive NeuroGlasses-Konversation schließen, wenn der Nutzer klar stoppen, abbrechen, sich verabschieden oder die Sitzung beenden will. Nicht für normale Unterhaltung verwenden.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("open_rokid_native_app"),
                description = "Native Rokid-App auf der Brille über Hi Rokid CXR-L öffnen. Für Rokid-Übersetzung, Galerie, Einstellungen, Helligkeit, Lautstärke, Teleprompter, Recorder oder andere Brillen-Apps verwenden. Musikplayer sind nicht unterstützt. Kamera nur öffnen, wenn der Nutzer ausdrücklich die native Kamera-App will; für schnelle Sichtfragen brillenfoto_aufnehmen nutzen. Für Navigation mit Ziel navigation_starten verwenden.",
                parameters = objectSchema(listOf("app"), mapOf(
                    "app" to mapOf(
                        "type" to "string",
                        "description" to "Zu öffnende native Brillen-App.",
                        "enum" to listOf(
                            "uebersetzer",
                            "navigation",
                            "kamera",
                            "galerie",
                            "einstellungen",
                            "bluetooth_einstellungen",
                            "systeminfo",
                            "helligkeit",
                            "lautstaerke",
                            "teleprompter",
                            "aufnahme",
                            "browser",
                            "start"
                        )
                    )
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("navigation"),
                description = "Navigation starten, Kandidat bestätigen oder Navigation stoppen. Eine laufende Zielauswahl wird mit action=confirm bestätigt.",
                parameters = objectSchema(listOf("action"), mapOf(
                    "action" to mapOf(
                        "type" to "string",
                        "description" to "Navigationsaktion.",
                        "enum" to listOf("start", "confirm", "stop")
                    ),
                    "target" to stringProp("Zieladresse, Ortsname oder korrigiertes Ziel bei action=start/confirm."),
                    "selection" to stringProp("Auswahl wie 1, erste, zweite oder Kandidatenname bei action=confirm."),
                    "mode" to mapOf(
                        "type" to "string",
                        "description" to "Routenmodus. Standard oeffi; zu_fuss nur bei ausdrücklichem Gehen/Fußweg.",
                        "enum" to listOf("oeffi", "zu_fuss")
                    )
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("get_gps_location"),
                description = "Letzten bekannten GPS-/Netzwerkstandort des Telefons lesen und Koordinaten für freihändige Nutzung zurückgeben.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("get_weather"),
                description = "Kurzen Wetterbericht für aktuellen GPS-Standort oder genannten Ort abrufen, ohne Browser zu öffnen.",
                parameters = objectSchema(emptyList(), mapOf("location" to stringProp("Optionale Stadt/Ort; leer lassen für aktuellen GPS-Standort.")))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("web_search"),
                description = "Websuchergebnisse und Ausschnitte für aktuelle Fragen abrufen, damit der Assistent freihändig antworten kann. Öffnet keinen Browser.",
                parameters = objectSchema(listOf("query"), mapOf("query" to stringProp("Suchanfrage.")))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("set_timer"),
                description = "Freihändig Android-Timer erstellen. Dauer selbst in Sekunden umrechnen und als positive Dezimalzahl übergeben.",
                parameters = objectSchema(listOf("seconds"), mapOf(
                    "seconds" to numericStringProp("Timerdauer in Sekunden als Dezimalziffern; muss größer als null sein."),
                    "label" to stringProp("Optionale Timer-Bezeichnung.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("create_reminder"),
                description = "Erinnerung erstellen. Wenn der Nutzer eine Fälligkeit nennt, exakte lokale epoch milliseconds selbst berechnen und als due_at_epoch_millis übergeben; die App interpretiert keinen natürlichen Zeittext.",
                parameters = objectSchema(listOf("text"), mapOf(
                    "text" to stringProp("Erinnerungstext."),
                    "when" to stringProp("Optionale lesbare Fälligkeitszeit nur zur Anzeige."),
                    "due_at_epoch_millis" to numericStringProp("Optionale exakte lokale Fälligkeit als Unix epoch milliseconds, als Dezimalziffern.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("list_reminders"),
                description = "Vom Assistenten gespeicherte Erinnerungen auflisten.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("create_calendar_event"),
                description = "Deutschsprachig: Lege einen Kalendertermin direkt an. Berechne Start/Ende selbst und uebergib epoch milliseconds als Dezimalziffern, wenn der Nutzer Zeiten nennt. Notizen nur für echte Notizen verwenden, nie Start/Ende dort wiederholen.",
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
                name = exposedToolName("open_app"),
                description = "Deutschsprachig: Öffne eine installierte App anhand von Paketname oder bekanntem App-Namen.",
                parameters = objectSchema(listOf("app"), mapOf("app" to stringProp("Paketname oder App-Name, z. B. Maps, WhatsApp, Kalender.")))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("share_text"),
                description = "Deutschsprachig: Teile oder diktiere Text über den Android-Teilen-Dialog.",
                parameters = objectSchema(listOf("text"), mapOf("text" to stringProp("Zu teilender Text.")))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("get_battery_status"),
                description = "Deutschsprachig: Lies den Akkustand und Ladezustand des Telefons aus.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("memory"),
                description = "Dauerhaftes Gedächtnis verwalten: stabile Fakten speichern, durchsuchen, auflisten oder löschen.",
                parameters = objectSchema(listOf("action"), mapOf(
                    "action" to mapOf(
                        "type" to "string",
                        "description" to "Gedächtnisaktion.",
                        "enum" to listOf("save", "search", "list", "delete")
                    ),
                    "content" to stringProp("Eigenständiger Erinnerungssatz bei action=save."),
                    "query" to stringProp("Suchfrage, Thema, ID oder Löschziel bei search/delete."),
                    "category" to stringProp("Optionale Kategorie, z. B. vorliebe, identitaet, routine, projekt, barrierefreiheit, reise oder allgemein."),
                    "tags" to stringProp("Optionale kommagetrennte Schlagwörter."),
                    "importance" to numericStringProp("Optionale Wichtigkeit 1-5; 5 bedeutet fast immer relevant."),
                    "limit" to numericStringProp("Optionale maximale Trefferzahl.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("saved_place"),
                description = "Gespeicherte Orte verwalten: speichern, suchen/listen, dorthin navigieren oder löschen. Speichern darf ohne Namen erfolgen; die App nutzt Koordinaten, Adresse oder aktuellen GPS-Standort und vergibt dann selbst einen Namen.",
                parameters = objectSchema(listOf("action"), mapOf(
                    "action" to mapOf(
                        "type" to "string",
                        "description" to "Ortsaktion.",
                        "enum" to listOf("save", "list", "navigate", "delete")
                    ),
                    "name" to stringProp("Optionaler Ortsname, ID oder Lösch-/Navigationsziel; bei save leer lassen, wenn nur der aktuelle Standort gespeichert werden soll."),
                    "query" to stringProp("Suchtext für action=list oder Alternative zu name."),
                    "description" to stringProp("Optionale Beschreibung oder Merkhilfe."),
                    "address" to stringProp("Optionale Adresse oder Ortsangabe zum Auflösen."),
                    "latitude" to numericStringProp("Optionaler Breitengrad als Dezimalzahl."),
                    "longitude" to numericStringProp("Optionaler Längengrad als Dezimalzahl."),
                    "mode" to mapOf(
                        "type" to "string",
                        "description" to "Routenmodus. Standard oeffi; zu_fuss nur bei ausdrücklichem Gehen/Fußweg.",
                        "enum" to listOf("oeffi", "zu_fuss")
                    ),
                    "limit" to numericStringProp("Optionale maximale Trefferzahl.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("get_agenda"),
                description = "Bevorstehende Kalendertermine vom Telefon lesen. Für Agenda, Tagesplan, Kalenderabfrage, nächsten Termin, heute, morgen oder diese Woche verwenden. Benötigt READ_CALENDAR.",
                parameters = objectSchema(emptyList(), mapOf(
                    "range" to mapOf(
                        "type" to "string",
                        "description" to "Zeitraum, wenn keine exakten Millisekunden übergeben werden.",
                        "enum" to listOf("heute", "morgen", "naechste_24_stunden", "woche")
                    ),
                    "start_epoch_millis" to numericStringProp("Optionaler exakter Start des Zeitraums als Unix epoch milliseconds."),
                    "end_epoch_millis" to numericStringProp("Optionales exaktes Ende des Zeitraums als Unix epoch milliseconds."),
                    "query" to stringProp("Optionaler Textfilter für Titel oder Ort.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("find_nearby_places"),
                description = "Nahe reale Orte über aktuellen GPS-Standort und OpenStreetMap finden, z. B. Apotheke, WC, Bankomat, Supermarkt, Haltestelle, Restaurant, Café, Krankenhaus, Trinkwasser, Parkplatz, Hotel oder freier Suchtext.",
                parameters = objectSchema(listOf("category"), mapOf(
                    "category" to stringProp("Ortskategorie oder Suchtext."),
                    "radius_meters" to numericStringProp("Optionaler Radius in Metern, Standard 1000."),
                    "limit" to numericStringProp("Optionale maximale Trefferzahl.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("get_transit_departures"),
                description = "Nahe österreichische Öffi-Haltestellen oder eine genannte Haltestelle finden und kommende OEBB/HAFAS-Abfahrten lesen. Für Zug, Bim, Bus, U-Bahn, Abfahrtstafel oder nächste Abfahrten verwenden.",
                parameters = objectSchema(emptyList(), mapOf(
                    "stop" to stringProp("Optionaler Haltestellen-/Stationsname; leer lassen für nächste Haltestellen per GPS."),
                    "radius_meters" to numericStringProp("Optionaler Suchradius in Metern, Standard 1200."),
                    "limit" to numericStringProp("Optionale maximale Zahl an Abfahrten.")
                ))
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("open_accessibility_settings"),
                description = "Deutschsprachig: Öffne Bedienungshilfen, falls der Nutzer Freisprech- oder Barrierefreiheitsrechte einrichten will.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("snap_glasses_photo"),
                description = "Frisches Rokid-Brillenfoto für den Assistenten aufnehmen, ohne die native Kameraoberfläche zu öffnen. Für schnelle Sichtfragen nutzen: was zu sehen ist, aktuelle Szene, Objekte, Personen, Hände, Anzahl, Farben, sichtbarer Text, Übersetzung, QR-Codes, Barcodes, Produktlabels oder alles, was Sehen braucht. Sichtfragen nicht aus Erinnerung beantworten, wenn kein Bild angehängt ist.",
                parameters = objectSchema(emptyList(), emptyMap())
            )),
            AssistantTool(function = ToolFunction(
                name = exposedToolName("chat"),
                description = "Persistente Chats verwalten: neu, auflisten, wechseln, umbenennen, verlassen oder löschen.",
                parameters = objectSchema(listOf("action"), mapOf(
                    "action" to mapOf(
                        "type" to "string",
                        "description" to "Chat-Aktion.",
                        "enum" to listOf("new", "list", "switch", "rename", "leave", "delete")
                    ),
                    "title" to stringProp("Chat-Titel für new, switch oder rename."),
                    "chat_id" to stringProp("Optionale Chat-ID für switch, rename oder delete.")
                ))
            ))
        )

        return tools
            .map(::compactAssistantTool)
            .filter { tool ->
                when (internalToolName(tool.function.name)) {
                    "chat", "new_chat", "list_chats", "switch_chat", "rename_chat", "leave_chat", "delete_chat" -> includeChatTools
                    "snap_glasses_photo" -> allowGlassesPhotoTool
                    else -> true
                }
            }
    }

    private fun compactAssistantTool(tool: AssistantTool): AssistantTool {
        val internalName = internalToolName(tool.function.name)
        return tool.copy(
            function = tool.function.copy(
                description = compactToolDescription(internalName, tool.function.description),
                parameters = compactToolParameters(tool.function.parameters)
            )
        )
    }

    private fun compactToolDescription(internalName: String, fallback: String): String = when (internalName) {
        "place_phone_call" -> "Anrufen; Kontaktname oder Telefonnummer genügt."
        "send_sms" -> "SMS senden; Kontaktname/Nummer und Nachricht."
        "find_contact_phone" -> "Telefonnummer eines Kontakts suchen."
        "stop_conversation" -> "Konversation nur bei klarem Stopp, Abbruch oder Abschied beenden."
        "open_rokid_native_app" -> "Native Rokid-App öffnen; nicht für Zielnavigation oder schnelle Sichtfragen."
        "navigation" -> "Navigation starten, bestätigen oder stoppen."
        "start_navigation" -> "Österreichisches Ziel auflösen und Navigation starten; Standardmodus oeffi."
        "confirm_navigation_destination" -> "Offenes Navigationsziel per Nummer, Name oder korrigierter Adresse bestätigen."
        "stop_navigation" -> "Aktive Hintergrundnavigation stoppen."
        "open_rokid_translator" -> "Native Rokid-Echtzeitübersetzung öffnen."
        "get_gps_location" -> "Letzten bekannten Telefonstandort abrufen."
        "get_weather" -> "Wetter für aktuellen Standort oder genannten Ort abrufen."
        "web_search" -> "Aktuelle Webinfos abrufen; keinen Browser öffnen."
        "set_timer" -> "Android-Timer stellen; Dauer vorher in Sekunden umrechnen."
        "create_reminder" -> "Erinnerung speichern; genannte Fälligkeit als epoch millis berechnen."
        "list_reminders" -> "Gespeicherte Erinnerungen auflisten."
        "create_calendar_event" -> "Kalendertermin direkt erstellen; Zeiten in Felder, nicht in Notizen."
        "open_app" -> "Installierte App per Paketname oder App-Name öffnen."
        "share_text" -> "Text über den Android-Teilen-Dialog teilen."
        "get_battery_status" -> "Akkustand und Ladezustand abrufen."
        "memory" -> "Dauerhaftes Gedächtnis speichern, suchen, listen oder löschen."
        "remember_memory" -> "Stabile Nutzerfakten, Vorlieben oder Kontext speichern."
        "search_memory" -> "Gespeichertes semantisches Gedächtnis durchsuchen."
        "list_memories" -> "Gespeicherte Erinnerungen optional gefiltert auflisten."
        "forget_memory" -> "Gespeicherte Erinnerung nur auf ausdrücklichen Wunsch löschen."
        "saved_place" -> "Gespeicherte Orte speichern, suchen, navigieren oder löschen; save darf ohne Namen aktuellen GPS-Ort nutzen."
        "save_place" -> "Ort dauerhaft speichern; Name optional, GPS oder Adresse nutzen."
        "list_saved_places" -> "Gespeicherte Orte optional gefiltert auflisten."
        "navigate_saved_place" -> "Navigation zu gespeichertem Ort per ID oder Name starten."
        "forget_saved_place" -> "Gespeicherten Ort nur auf ausdrücklichen Wunsch löschen."
        "get_agenda" -> "Bevorstehende Kalendertermine lesen."
        "find_nearby_places" -> "Nahe reale Orte per GPS und OpenStreetMap finden."
        "get_transit_departures" -> "Nahe oder genannte Öffi-Haltestelle finden und Abfahrten lesen."
        "open_accessibility_settings" -> "Android-Bedienungshilfen öffnen."
        "snap_glasses_photo" -> "Frisches Brillenfoto für Sichtfragen aufnehmen."
        "chat" -> "Persistente Chats verwalten."
        "new_chat" -> "Neuen persistenten Chat erstellen und aktivieren."
        "list_chats" -> "Persistente Chats mit IDs und aktivem Chat auflisten."
        "switch_chat" -> "Zu persistentem Chat per ID oder Titel wechseln."
        "rename_chat" -> "Aktiven oder angegebenen Chat umbenennen."
        "leave_chat" -> "Persistenten Chat verlassen und flüchtig weiterreden."
        "delete_chat" -> "Persistenten Chat nur auf ausdrücklichen Wunsch löschen."
        else -> fallback.collapseWhitespace().take(160)
    }

    private fun compactToolParameters(parameters: Map<String, Any>): Map<String, Any> {
        val rawProperties = parameters["properties"] as? Map<*, *> ?: return parameters
        if (rawProperties.isEmpty()) return parameters

        val compactProperties = rawProperties.mapNotNull { (rawName, rawProperty) ->
            val name = rawName as? String ?: return@mapNotNull null
            val property = rawProperty as? Map<*, *> ?: return@mapNotNull null
            name to compactToolProperty(property)
        }.toMap()

        return parameters.toMutableMap().apply {
            this["properties"] = compactProperties
        }
    }

    private fun compactToolProperty(property: Map<*, *>): Map<String, Any> {
        val compact = mutableMapOf<String, Any>()
        (property["type"] as? String)?.let { compact["type"] = it }
        property["enum"]?.let { compact["enum"] = it }
        return compact
    }

    private fun Map<String, String>.firstArg(vararg names: String): String =
        names.firstNotNullOfOrNull { name -> this[name]?.trim()?.takeIf { it.isNotBlank() } }.orEmpty()

    private fun normalizedToolAction(action: String): String =
        action.trim().lowercase(Locale.ROOT)

    private fun executeNavigationTool(args: Map<String, String>): String {
        val action = normalizedToolAction(args.firstArg("action", "operation")).ifBlank {
            if (args.firstArg("selection").isNotBlank()) "confirm" else "start"
        }
        val target = args.firstArg("target", "destination", "address")
        val selection = args.firstArg("selection", "choice")
        val mode = args.firstArg("mode")

        return when (action) {
            "stop", "cancel", "end" -> stopOsmNavigation()
            "confirm", "choose", "select" -> confirmOsmNavigationDestination(
                selection.ifBlank { target },
                target,
                mode
            )
            else -> startOsmNavigation(target, mode)
        }
    }

    private fun executeMemoryTool(args: Map<String, String>): String {
        val action = normalizedToolAction(args.firstArg("action", "operation")).ifBlank {
            when {
                args.firstArg("content").isNotBlank() -> "save"
                args.firstArg("query", "selection", "id").isNotBlank() -> "search"
                else -> "list"
            }
        }
        val query = args.firstArg("query", "selection", "id", "content")

        return when (action) {
            "save", "remember", "add" -> rememberMemory(
                args.firstArg("content"),
                args.firstArg("category"),
                args.firstArg("tags"),
                args.firstArg("importance")
            )
            "delete", "remove", "forget" -> forgetMemory(query)
            "search", "find" -> searchMemory(query, args.firstArg("category"), args.firstArg("limit"))
            else -> listMemories(args.firstArg("category"), args.firstArg("limit"))
        }
    }

    private fun executeSavedPlaceTool(args: Map<String, String>): String {
        val action = normalizedToolAction(args.firstArg("action", "operation")).ifBlank {
            when {
                args.firstArg("address", "latitude", "longitude").isNotBlank() -> "save"
                args.firstArg("name", "name_or_id", "id").isNotBlank() -> "navigate"
                else -> "list"
            }
        }
        val nameOrQuery = args.firstArg("name", "name_or_id", "id", "query")

        return when (action) {
            "save", "add" -> savePlace(
                args.firstArg("name").ifBlank { nameOrQuery },
                args.firstArg("description"),
                args.firstArg("address"),
                args.firstArg("latitude"),
                args.firstArg("longitude")
            )
            "navigate", "go", "route" -> navigateSavedPlace(nameOrQuery, args.firstArg("mode"))
            "delete", "remove", "forget" -> forgetSavedPlace(nameOrQuery)
            else -> listSavedPlaces(args.firstArg("query").ifBlank { nameOrQuery }, args.firstArg("limit"))
        }
    }

    private fun executeChatTool(args: Map<String, String>): String {
        val action = normalizedToolAction(args.firstArg("action", "operation")).ifBlank { "list" }
        return when (action) {
            "new", "create", "start" -> {
                val session = createChatSession(args.firstArg("title"))
                "Neuen Chat '${session.title}' (${session.id}) erstellt und dorthin gewechselt."
            }
            "switch", "open", "select" -> switchChat(args.firstArg("chat_id", "id"), args.firstArg("title"))
            "rename" -> renameChat(args.firstArg("chat_id", "id"), args.firstArg("title"))
            "leave", "exit" -> leaveChat()
            "delete", "remove" -> deleteChat(args.firstArg("chat_id", "id"))
            else -> listChats()
        }
    }

    private fun executeAssistantTool(
        toolCall: ToolCall,
        instruction: String,
        notifyListener: Boolean = true
    ): String {
        val args = decodeToolArguments(toolCall.function.arguments)
        val toolName = internalToolName(toolCall.function.name)
        if (notifyListener) {
            listener?.onAssistantToolCall(toolName, args)?.let { return it }
        }

        fun arg(name: String): String = args[name].orEmpty()

        return when (toolName) {
            "place_phone_call" -> placePhoneCall(arg("recipient"))
            "send_sms" -> sendSms(arg("recipient"), arg("message"))
            "find_contact_phone" -> findContactPhone(arg("recipient"))
            "stop_conversation" -> "Konversation beendet."
            "open_rokid_native_app" -> openRokidNativeApp(arg("app"))
            "navigation" -> executeNavigationTool(args)
            "start_navigation" -> startOsmNavigation(arg("destination"), arg("mode"))
            "confirm_navigation_destination" -> confirmOsmNavigationDestination(arg("selection"), arg("destination"), arg("mode"))
            "stop_navigation" -> stopOsmNavigation()
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
            "open_app" -> openApp(arg("app"))
            "share_text" -> shareText(arg("text"))
            "get_battery_status" -> getBatteryStatus()
            "memory" -> executeMemoryTool(args)
            "remember_memory" -> rememberMemory(arg("content"), arg("category"), arg("tags"), arg("importance"))
            "search_memory" -> searchMemory(arg("query"), arg("category"), arg("limit"))
            "list_memories" -> listMemories(arg("category"), arg("limit"))
            "forget_memory" -> forgetMemory(arg("selection"))
            "saved_place" -> executeSavedPlaceTool(args)
            "save_place" -> savePlace(arg("name"), arg("description"), arg("address"), arg("latitude"), arg("longitude"))
            "list_saved_places" -> listSavedPlaces(arg("query"), arg("limit"))
            "navigate_saved_place" -> navigateSavedPlace(arg("name_or_id"), arg("mode"))
            "forget_saved_place" -> forgetSavedPlace(arg("name_or_id"))
            "get_agenda" -> getAgenda(
                arg("range"),
                arg("start_epoch_millis").toLongOrNull(),
                arg("end_epoch_millis").toLongOrNull(),
                arg("query")
            )
            "find_nearby_places" -> findNearbyPlaces(
                arg("category"),
                arg("radius_meters").toIntOrNull(),
                arg("limit").toIntOrNull()
            )
            "get_transit_departures" -> getTransitDepartures(
                arg("stop"),
                arg("radius_meters").toIntOrNull(),
                arg("limit").toIntOrNull()
            )
            "open_accessibility_settings" -> launchIntent(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS), "Bedienungshilfen")
            "chat" -> executeChatTool(args)
            "new_chat" -> {
                val session = createChatSession(arg("title"))
                "Neuen Chat '${session.title}' (${session.id}) erstellt und dorthin gewechselt."
            }
            "list_chats" -> listChats()
            "switch_chat" -> switchChat(arg("chat_id"), arg("title"))
            "rename_chat" -> renameChat(arg("chat_id"), arg("title"))
            "leave_chat" -> leaveChat()
            "delete_chat" -> deleteChat(arg("chat_id"))
            else -> "Unbekanntes Werkzeug: ${toolCall.function.name}"
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
        toolName in setOf("navigation", "start_navigation", "confirm_navigation_destination", "navigate_saved_place") ||
            navigationActuallyStarted(result)

    private fun navigationActuallyStarted(result: String): Boolean =
        result.startsWith("Navigation gestartet", ignoreCase = true) ||
            result.startsWith("Navigation mode started", ignoreCase = true)
    
    private fun buildToolResultFallback(results: List<String>): String {
        if (results.isEmpty()) return ""
        val last = results.last()
        val toolName = internalToolName(last.substringBefore(":").trim())
        val lastResult = last.substringAfter(":").trim()
        return when {
            toolName == "web_search" ->
                "Ich habe die Websuche ausgeführt, aber danach keinen KI-Antworttext bekommen. Hier sind die Rohfunde:\n${lastResult.take(2600)}"
            lastResult.startsWith("Gestartet:", ignoreCase = true) -> "Erledigt: ${lastResult.removePrefix("Gestartet:").trim()}"
            lastResult.startsWith("Started ", ignoreCase = true) -> "Erledigt: ${lastResult.removePrefix("Started ").removeSuffix(" hands-free.")} gestartet."
            lastResult.startsWith("Angefordert:", ignoreCase = true) -> lastResult
            lastResult.startsWith("Requested ", ignoreCase = true) -> lastResult
            lastResult.startsWith("Konnte", ignoreCase = true) -> "Das hat leider nicht geklappt: $lastResult"
            lastResult.startsWith("Could not", ignoreCase = true) -> "Das hat leider nicht geklappt: $lastResult"
            lastResult.contains("Berechtigung", ignoreCase = true) -> "Dafür fehlt noch eine Berechtigung: $lastResult"
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
            "Wenn die Frage nach Anzahl, Farbe, Text, Übersetzung, QR-Code, Barcode, Produktlabel oder Details fragt, antworte direkt aus dem Bild. " +
            "Bei QR-Codes oder Barcodes gib den sichtbaren/erkennbaren Inhalt, Link oder Codewert aus; wenn er nicht sicher lesbar ist, sag das knapp. " +
            "Wenn ein Detail nicht sicher erkennbar ist, sag das knapp. Nutzerfrage: $instruction"

    private fun launchIntent(intent: Intent, label: String): String = try {
        val launchIntent = Intent(intent).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        if (context is Activity) {
            context.startActivity(launchIntent)
            "Gestartet: $label."
        } else {
            launchIntentFromBackground(launchIntent, label)
        }
    } catch (e: Exception) {
        Log.e(appTag, "Could not start $label", e)
        val lockedSuffix = if (isDeviceLocked()) " Das Telefon ist gesperrt; entsperre es und versuche es erneut, falls Android die Ziel-App blockiert hat." else ""
        "Konnte $label nicht starten: ${e.message}.$lockedSuffix"
    }

    private fun launchIntentFromBackground(intent: Intent, label: String): String {
        val sendOptions = backgroundActivityLaunchOptions("setPendingIntentBackgroundActivityStartMode")
        val creatorOptions = backgroundActivityLaunchOptions("setPendingIntentCreatorBackgroundActivityStartMode")
        val requestCode = (System.currentTimeMillis() and Int.MAX_VALUE.toLong()).toInt()
        val pendingIntent = PendingIntent.getActivity(
            context,
            requestCode,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            creatorOptions
        )
        val notificationPosted = showBackgroundLaunchNotification(label, pendingIntent)

        return try {
            pendingIntent.send(context, 0, null, null, null, null, sendOptions)
            if (notificationPosted) {
                "Angefordert: $label über Android. Zusätzlich wurde eine Startbenachrichtigung angezeigt, falls Android das Öffnen im Hintergrund blockiert."
            } else {
                "Angefordert: $label über Android. Falls nichts geöffnet wird, hat Android diesen Hintergrundstart blockiert."
            }
        } catch (e: PendingIntent.CanceledException) {
            Log.e(appTag, "Could not send pending intent for $label", e)
            "Konnte $label nicht starten: ${e.message}."
        } catch (e: Exception) {
            Log.e(appTag, "Could not request $label", e)
            "Konnte $label nicht starten: ${e.message}."
        }
    }

    private fun backgroundActivityLaunchOptions(methodName: String): Bundle? {
        val options = ActivityOptions.makeBasic()
        val mode = backgroundActivityStartMode() ?: return options.toBundle()
        runCatching {
            ActivityOptions::class.java
                .getMethod(methodName, Int::class.javaPrimitiveType)
                .invoke(options, mode)
        }.onFailure {
            Log.d(appTag, "ActivityOptions.$methodName unavailable on API ${Build.VERSION.SDK_INT}")
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

    private fun showBackgroundLaunchNotification(label: String, pendingIntent: PendingIntent): Boolean {
        if (!canPostNotifications()) return false
        val manager = context.getSystemService(NotificationManager::class.java) ?: return false
        createToolLaunchNotificationChannel(manager)

        val notification = Notification.Builder(context, TOOL_LAUNCH_CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("$label öffnen")
            .setContentText("Android kann dieses Tippen für Starts im Hintergrund verlangen.")
            .setCategory(Notification.CATEGORY_CALL)
            .setVisibility(Notification.VISIBILITY_PUBLIC)
            .setPriority(Notification.PRIORITY_MAX)
            .setAutoCancel(true)
            .setTimeoutAfter(TOOL_LAUNCH_NOTIFICATION_TIMEOUT_MS)
            .setContentIntent(pendingIntent)
            .setFullScreenIntent(pendingIntent, true)
            .build()

        return runCatching {
            manager.notify(TOOL_LAUNCH_NOTIFICATION_ID, notification)
            true
        }.getOrElse {
            Log.w(appTag, "Could not post launch notification for $label", it)
            false
        }
    }

    private fun canPostNotifications(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU ||
            hasPermission(Manifest.permission.POST_NOTIFICATIONS)

    private fun createToolLaunchNotificationChannel(manager: NotificationManager) {
        if (manager.getNotificationChannel(TOOL_LAUNCH_CHANNEL_ID) != null) return
        val channel = NotificationChannel(
            TOOL_LAUNCH_CHANNEL_ID,
            "NeuroGlasses Werkzeugstarts",
            NotificationManager.IMPORTANCE_HIGH
        )
        channel.lockscreenVisibility = Notification.VISIBILITY_PUBLIC
        manager.createNotificationChannel(channel)
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
            target.number != null && target.displayName != null -> "Gefunden: ${target.displayName}: ${target.number}."
            target.number != null -> "Der Empfänger ist bereits eine Telefonnummer: ${target.number}."
            target.contactPermissionMissing -> "Die READ_CONTACTS-Berechtigung ist erforderlich, bevor ich $recipient im Telefonbuch suchen kann."
            else -> "Ich konnte keine Telefonnummer für $recipient finden."
        }
    }

    @SuppressLint("MissingPermission")
    private fun placePhoneCall(recipient: String): String {
        val target = resolvePhoneTarget(recipient)
        val phone = target.number
        if (phone.isNullOrBlank()) {
            return if (target.contactPermissionMissing) {
                "Die READ_CONTACTS-Berechtigung ist erforderlich, bevor ich $recipient finden kann. Gib eine Telefonnummer an oder erteile die Kontakte-Berechtigung."
            } else {
                "Ich konnte keine Telefonnummer für $recipient finden."
            }
        }

        val label = target.displayName?.let { "Anruf an $it" } ?: "Anruf"
        if (hasPermission(Manifest.permission.CALL_PHONE)) {
            val callUri = Uri.parse("tel:${Uri.encode(phone)}")
            val telecomManager = context.getSystemService(TelecomManager::class.java)
            if (telecomManager != null) {
                runCatching {
                    telecomManager.placeCall(callUri, Bundle.EMPTY)
                }.onSuccess {
                    return "Gestartet: $label."
                }.onFailure { error ->
                    Log.e(appTag, "TelecomManager.placeCall failed", error)
                }
            }

            return launchIntent(Intent(Intent.ACTION_CALL, callUri), label)
        }

        val dialResult = launchIntent(Intent(Intent.ACTION_DIAL, Uri.parse("tel:${Uri.encode(phone)}")), "Wählhilfe für $label")
        return if (dialResult.startsWith("Could not", ignoreCase = true) || dialResult.startsWith("Konnte", ignoreCase = true)) {
            "CALL_PHONE-Berechtigung fehlt und ich konnte die Wählhilfe nicht öffnen: $dialResult"
        } else {
            "CALL_PHONE-Berechtigung fehlt, daher habe ich die Wählhilfe für ${target.displayName ?: phone} geöffnet."
        }
    }

    private fun sendSms(recipient: String, message: String): String {
        if (message.isBlank()) return "Ich brauche eine Nachricht, bevor ich eine SMS senden kann."
        val target = resolvePhoneTarget(recipient)
        val phone = target.number
        if (phone.isNullOrBlank()) {
            return if (target.contactPermissionMissing) {
                "Die READ_CONTACTS-Berechtigung ist erforderlich, bevor ich $recipient finden kann. Gib eine Telefonnummer an oder erteile die Kontakte-Berechtigung."
            } else {
                "Ich konnte keine Telefonnummer für $recipient finden."
            }
        }

        if (!hasPermission(Manifest.permission.SEND_SMS)) {
            return openSmsComposer(phone, message, "SEND_SMS-Berechtigung fehlt, daher habe ich stattdessen den SMS-Composer geöffnet.")
        }

        return try {
            val smsManager = context.getSystemService(SmsManager::class.java)
            val parts = smsManager.divideMessage(message)
            if (parts.size > 1) {
                smsManager.sendMultipartTextMessage(phone, null, parts, null, null)
            } else {
                smsManager.sendTextMessage(phone, null, message, null, null)
            }
            "SMS an ${target.displayName ?: phone} gesendet."
        } catch (e: Exception) {
            Log.e(appTag, "Could not send SMS directly", e)
            openSmsComposer(phone, message, "Konnte SMS nicht direkt senden (${e.message}); habe stattdessen den SMS-Composer geöffnet.")
        }
    }

    private fun openSmsComposer(phone: String, message: String, prefix: String): String {
        val result = launchIntent(
            Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:${Uri.encode(phone)}"))
                .putExtra("sms_body", message),
            "SMS-Composer"
        )
        return if (result.startsWith("Could not", ignoreCase = true) || result.startsWith("Konnte", ignoreCase = true)) "$prefix $result" else prefix
    }

    private fun openRokidTranslator(): String = openRokidNativeApp("translator")

    private fun startOsmNavigation(destination: String, mode: String): String {
        val cleanMode = normalizeNavigationMode(mode)
        findSavedPlaceForNavigation(destination)?.let { place ->
            return startNavigationToSavedPlace(place, cleanMode)
        }

        val resolution = navigationDestinationResolver.resolve(destination, cleanMode)
        val bestEffortResolution = if (
            resolution.status == NavigationDestinationResolver.Status.NEEDS_CONFIRMATION &&
            resolution.candidates.isNotEmpty()
        ) {
            navigationDestinationResolver.confirm(
                selection = "1",
                modeOverride = resolution.mode.ifBlank { cleanMode }
            ).takeIf { it.status == NavigationDestinationResolver.Status.CONFIRMED } ?: resolution
        } else {
            resolution
        }
        return startNavigationFromResolution(bestEffortResolution, cleanMode)
    }

    private fun confirmOsmNavigationDestination(selection: String, destination: String, mode: String): String {
        val resolution = navigationDestinationResolver.confirm(
            selection = selection,
            refinedDestination = destination.takeIf { it.isNotBlank() },
            modeOverride = mode.takeIf { it.isNotBlank() }?.let(::normalizeNavigationMode)
        )
        return startNavigationFromResolution(resolution, normalizeNavigationMode(mode))
    }

    private fun startNavigationFromResolution(
        resolution: NavigationDestinationResolver.Result,
        fallbackMode: String
    ): String {
        return when (resolution.status) {
            NavigationDestinationResolver.Status.CONFIRMED -> {
                val selected = resolution.selected ?: return resolution.message
                RokidConnectionService.startNavigation(
                    context = context,
                    destination = selected.shortLabel,
                    mode = normalizeNavigationMode(resolution.mode.ifBlank { fallbackMode }),
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

    private fun normalizeNavigationMode(mode: String): String =
        when (mode.trim().lowercase(Locale.ROOT)) {
            "zu_fuss", "zu_fuss.", "zu fuss", "zu fuß", "zu_fuß", "fuss", "fuß", "foot", "walk", "walking", "pedestrian" -> "foot"
            "oeffi", "öffi", "offi", "oepnv", "öpnv", "oeffentlich", "öffentlich", "public_transit", "transit", "default", "" -> "public_transit"
            else -> mode
        }

    private fun openRokidNavigation(destination: String): String {
        val launchResult = RokidHostConnection.openGlassApp(rokidNavigationTargets())
        if (!launchResult.success) return launchResult.detail

        val cleanDestination = destination.trim()
        return if (cleanDestination.isNotBlank()) {
            val commandResult = RokidHostConnection.sendNavigationDestination(cleanDestination)
            if (commandResult.success) {
                "${launchResult.label} auf der Brille geöffnet und Ziel gesendet: $cleanDestination."
            } else {
                "${launchResult.label} auf der Brille geöffnet, aber Ziel '$cleanDestination' konnte nicht gesendet werden: ${commandResult.detail}"
            }
        } else {
            "${launchResult.label} auf der Brille geöffnet."
        }
    }

    private fun openRokidNativeApp(app: String): String {
        val normalized = app.trim().lowercase(Locale.ROOT)
        val targets = when {
            normalized in setOf("translator", "translate", "translation", "übersetzer", "uebersetzer", "uebersetzer", "übersetzung", "uebersetzung", "rt translation", "live translation") ->
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
            normalized in setOf("bluetooth_settings", "bluetooth_einstellungen", "bluetooth", "bt", "bluetooth settings", "bluetooth-einstellungen") ->
                spriteLauncherTarget("Rokid Bluetooth Settings", ".setting.bluetooth.SettingBluetoothActivity")
            normalized in setOf("system_info", "systeminfo", "system info", "info", "geräteinfo", "geraeteinfo") ->
                spriteLauncherTarget("Rokid System Info", ".setting.info.SettingSystemInfoActivity")
            normalized in setOf("brightness", "helligkeit", "display brightness") ->
                spriteLauncherTarget("Rokid Brightness", ".page.brightness.SettingBrightnessActivity")
            normalized in setOf("volume", "lautstärke", "lautstaerke", "sound", "audio volume") ->
                spriteLauncherTarget("Rokid Volume", ".page.volume.SettingVolumeActivity")
            normalized in setOf("teleprompter", "wordtips", "word tips", "worttipps", "spickzettel") ->
                spriteLauncherTarget("Rokid Word Tips", ".page.wordtips.WordTipsPageActivity")
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
            return "Unbekannte native Rokid-App '$app'. Unterstützt: uebersetzer, navigation, kamera, galerie, einstellungen, bluetooth_einstellungen, systeminfo, helligkeit, lautstaerke, teleprompter, aufnahme, browser, start."
        }

        val result = RokidHostConnection.openGlassApp(targets)
        return if (result.success) {
            "${result.label} auf der Brille geöffnet."
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
            return "Standort-Berechtigung ist erforderlich, bevor ich GPS freihändig lesen kann."
        }
        return try {
            val location = readLastKnownLocation()
            if (location == null) {
                "Noch kein letzter bekannter Standort verfügbar."
            } else {
                "Aktueller Standort: Breitengrad=${location.latitude}, Längengrad=${location.longitude}, Genauigkeit=${location.accuracyMeters}m."
            }
        } catch (e: Exception) {
            "Konnte Standort nicht lesen: ${e.message}"
        }
    }

    private fun getWeather(location: String): String {
        val query = if (location.isNotBlank()) {
            location
        } else {
            readLastKnownLocation()?.let { "${it.latitude},${it.longitude}" }.orEmpty()
        }
        if (query.isBlank()) return "Ich brauche einen Ort oder GPS-Berechtigung, um Wetter abzurufen."
        return fetchText("https://wttr.in/${URLEncoder.encode(query, "UTF-8")}?format=3", "Wetter")
    }

    private fun webSearch(query: String): String {
        val cleanQuery = query.trim()
        if (cleanQuery.isBlank()) return "Ich brauche einen Suchbegriff für die Websuche."

        val sections = mutableListOf<String>()
        val instantAnswer = fetchDuckDuckGoInstantAnswer(cleanQuery)
        if (instantAnswer.isNotBlank()) {
            sections += "Direktantwort:\n$instantAnswer"
        }

        val searchResults = fetchDuckDuckGoSearchResults(cleanQuery)
        if (searchResults.isNotEmpty()) {
            sections += "Web-Ergebnisse für \"$cleanQuery\":\n${formatWebSearchResults(searchResults)}"
        }

        return if (sections.isEmpty()) {
            "Ich konnte keine Web-Ergebnisse für \"$cleanQuery\" abrufen."
        } else {
            sections.joinToString("\n\n").take(WEB_SEARCH_MAX_OUTPUT_CHARS)
        }
    }

    private fun fetchDuckDuckGoInstantAnswer(query: String): String {
        val url = "https://api.duckduckgo.com/?q=${URLEncoder.encode(query, "UTF-8")}&format=json&no_html=1&skip_disambig=1"
        val raw = fetchText(url, "DuckDuckGo-Direktantwort", maxChars = 120_000)
        if (raw.startsWith("Could not", ignoreCase = true) || raw.startsWith("Konnte", ignoreCase = true)) {
            Log.w(appTag, raw)
            return ""
        }

        val parsed = runCatching { gson.fromJson(raw, Map::class.java) as Map<*, *> }.getOrNull() ?: return ""
        val heading = parsed["Heading"]?.toString().orEmpty()
        val abstract = parsed["AbstractText"]?.toString().orEmpty()
        val related = (parsed["RelatedTopics"] as? List<*>)?.firstOrNull()?.let { it as? Map<*, *> }?.get("Text")?.toString().orEmpty()
        return listOf(heading, abstract, related)
            .filter { it.isNotBlank() }
            .joinToString("\n")
            .take(1200)
    }

    private fun fetchDuckDuckGoSearchResults(query: String): List<WebSearchResult> {
        val url = "https://html.duckduckgo.com/html/?q=${URLEncoder.encode(query, "UTF-8")}"
        val html = fetchText(url, "DuckDuckGo-Suchergebnisse", maxChars = 180_000)
        if (html.startsWith("Could not", ignoreCase = true) || html.startsWith("Konnte", ignoreCase = true)) {
            Log.w(appTag, html)
            return emptyList()
        }

        val titleRegex = Regex("(?is)<a[^>]*class=[\"'][^\"']*result__a[^\"']*[\"'][^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>")
        val snippetRegex = Regex("(?is)<(?:a|div)[^>]*class=[\"'][^\"']*result__snippet[^\"']*[\"'][^>]*>(.*?)</(?:a|div)>")
        val matches = titleRegex.findAll(html).toList()
        return matches.mapIndexedNotNull { index, match ->
            val title = htmlToPlainText(match.groupValues.getOrNull(2).orEmpty())
            if (title.isBlank()) return@mapIndexedNotNull null

            val chunkStart = (match.range.last + 1).coerceAtMost(html.length)
            val chunkEnd = matches.getOrNull(index + 1)?.range?.first ?: (match.range.last + 4000).coerceAtMost(html.length)
            val chunk = html.substring(chunkStart, chunkEnd.coerceAtLeast(chunkStart))
            val snippet = snippetRegex.find(chunk)
                ?.groupValues
                ?.getOrNull(1)
                ?.let(::htmlToPlainText)
                .orEmpty()

            WebSearchResult(
                title = title,
                url = cleanDuckDuckGoResultUrl(match.groupValues.getOrNull(1).orEmpty()),
                snippet = snippet
            )
        }
            .distinctBy { result -> result.url.ifBlank { result.title } }
            .take(WEB_SEARCH_RESULT_LIMIT)
    }

    private fun formatWebSearchResults(results: List<WebSearchResult>): String =
        results.mapIndexed { index, result ->
            buildString {
                append("${index + 1}. ${result.title}")
                if (result.url.isNotBlank()) append("\n   ${result.url}")
                if (result.snippet.isNotBlank()) append("\n   ${result.snippet}")
            }
        }.joinToString("\n")

    private fun cleanDuckDuckGoResultUrl(rawUrl: String): String {
        val decoded = decodeHtmlEntities(rawUrl).trim()
        val absolute = when {
            decoded.startsWith("//") -> "https:$decoded"
            decoded.startsWith("/") -> "https://duckduckgo.com$decoded"
            else -> decoded
        }
        return runCatching {
            Uri.parse(absolute).getQueryParameter("uddg")?.takeIf { it.isNotBlank() } ?: absolute
        }.getOrDefault(absolute)
    }

    private fun htmlToPlainText(html: String): String =
        decodeHtmlEntities(
            html
                .replace(Regex("(?is)<script.*?</script>"), " ")
                .replace(Regex("(?is)<style.*?</style>"), " ")
                .replace(Regex("(?i)<br\\s*/?>"), " ")
                .replace(Regex("<[^>]+>"), " ")
        ).collapseWhitespace().trim()

    private fun decodeHtmlEntities(value: String): String =
        value
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&nbsp;", " ")
            .replace(Regex("&#x([0-9a-fA-F]+);")) { match ->
                match.groupValues.getOrNull(1)
                    ?.toIntOrNull(16)
                    ?.let(::codePointToString)
                    ?: match.value
            }
            .replace(Regex("&#(\\d+);")) { match ->
                match.groupValues.getOrNull(1)
                    ?.toIntOrNull()
                    ?.let(::codePointToString)
                    ?: match.value
            }

    private fun codePointToString(codePoint: Int): String =
        runCatching { String(Character.toChars(codePoint)) }.getOrDefault("")

    private fun fetchText(url: String, label: String, maxChars: Int = Int.MAX_VALUE): String = try {
        val request = Request.Builder()
            .url(url)
            .get()
            .header("User-Agent", WEB_SEARCH_USER_AGENT)
            .header("Accept-Language", "de,en;q=0.8")
            .build()
        getWebFetchClient().newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                "Konnte $label nicht abrufen: HTTP ${response.code}"
            } else {
                val text = response.body?.string()?.trim().orEmpty()
                if (maxChars == Int.MAX_VALUE) text else text.take(maxChars)
            }
        }
    } catch (e: Exception) {
        "Konnte $label nicht abrufen: ${e.message}"
    }

    private fun setTimer(seconds: Int, label: String): String {
        if (seconds <= 0) return "Die Timerdauer muss größer als null Sekunden sein."
        val intent = Intent(AlarmClock.ACTION_SET_TIMER)
            .putExtra(AlarmClock.EXTRA_LENGTH, seconds)
            .putExtra(AlarmClock.EXTRA_MESSAGE, label.ifBlank { "NeuroGlasses Timer" })
            .putExtra(AlarmClock.EXTRA_SKIP_UI, true)
        return launchIntent(intent, "${seconds}-Sekunden-Timer")
    }

    private fun createReminder(text: String, dueTime: String, dueAtMillis: Long?): String {
        if (text.isBlank()) return "Der Erinnerungstext darf nicht leer sein."
        val prefs = context.getSharedPreferences("assistant_reminders", Context.MODE_PRIVATE)
        val existing = prefs.getStringSet("items", emptySet()).orEmpty().toMutableSet()
        existing.add("${System.currentTimeMillis()}|$text|$dueTime|${dueAtMillis ?: ""}")
        prefs.edit().putStringSet("items", existing).apply()

        val savedMessage = "Erinnerung gespeichert: $text${if (dueTime.isNotBlank()) " ($dueTime)" else ""}."
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
                ?: return "Erinnerung gespeichert, aber Android AlarmManager ist nicht verfügbar."
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
            "Erinnerungsbenachrichtigung automatisch für $formattedTime geplant."
        } catch (e: Exception) {
            Log.e(appTag, "Could not schedule reminder notification", e)
            "Erinnerung gespeichert, aber ich konnte die Benachrichtigung nicht planen: ${e.message}"
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
        return if (reminders.isEmpty()) "Keine Erinnerungen gespeichert." else reminders.joinToString("\n")
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
            "youtube" to "com.google.android.youtube"
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
        val cleanTitle = title.trim()
        if (cleanTitle.isBlank()) return "Ich brauche einen Termintitel."
        val description = calendarEventDescription(notes)
        val beginMillis = startEpochMillis?.takeIf { it > 0 }
            ?: parseCalendarDateTimeMillis(start)
        val finishMillis = beginMillis?.let { startMillis ->
            val referenceDate = Instant.ofEpochMilli(startMillis).atZone(ZoneId.systemDefault()).toLocalDate()
            val parsedEndMillis = parseCalendarDateTimeMillis(end, referenceDate)
                ?.let { if (it <= startMillis) it + 24 * 60 * 60 * 1000L else it }
            endEpochMillis?.takeIf { it > startMillis }
                ?: parsedEndMillis
                ?: (startMillis + DEFAULT_CALENDAR_EVENT_DURATION_MILLIS)
        }

        if (beginMillis != null && finishMillis != null && hasPermission(Manifest.permission.WRITE_CALENDAR)) {
            createCalendarEventDirectly(cleanTitle, beginMillis, finishMillis, location, description)?.let {
                return it
            }
        }

        val intent = Intent(Intent.ACTION_INSERT)
            .setData(CalendarContract.Events.CONTENT_URI)
            .putExtra(CalendarContract.Events.TITLE, cleanTitle)
            .putExtra(CalendarContract.Events.EVENT_LOCATION, location)
            .putExtra(CalendarContract.Events.DESCRIPTION, description)
        beginMillis?.let { startMillis ->
            intent.putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, startMillis)
            intent.putExtra(CalendarContract.EXTRA_EVENT_END_TIME, finishMillis ?: (startMillis + DEFAULT_CALENDAR_EVENT_DURATION_MILLIS))
        }
        return launchIntent(intent, "Kalendertermin")
    }

    private fun createCalendarEventDirectly(
        title: String,
        beginMillis: Long,
        endMillis: Long,
        location: String,
        description: String
    ): String? {
        val calendar = writableCalendarCandidate() ?: return null
        return try {
            val values = ContentValues().apply {
                put(CalendarContract.Events.CALENDAR_ID, calendar.id)
                put(CalendarContract.Events.TITLE, title)
                put(CalendarContract.Events.DTSTART, beginMillis)
                put(CalendarContract.Events.DTEND, endMillis)
                put(CalendarContract.Events.EVENT_TIMEZONE, TimeZone.getDefault().id)
                location.trim().takeIf { it.isNotBlank() }?.let { put(CalendarContract.Events.EVENT_LOCATION, it) }
                description.takeIf { it.isNotBlank() }?.let { put(CalendarContract.Events.DESCRIPTION, it) }
            }
            val uri = context.contentResolver.insert(CalendarContract.Events.CONTENT_URI, values) ?: return null
            val eventId = ContentUris.parseId(uri)
            "Kalendertermin erstellt [${eventId}]: $title, ${formatAgendaTime(beginMillis, endMillis, allDay = false)} (${calendar.displayName})."
        } catch (e: Exception) {
            Log.e(appTag, "Could not create calendar event directly", e)
            null
        }
    }

    private fun writableCalendarCandidate(): WritableCalendarCandidate? {
        if (!hasPermission(Manifest.permission.READ_CALENDAR) && !hasPermission(Manifest.permission.WRITE_CALENDAR)) return null
        val projection = arrayOf(
            CalendarContract.Calendars._ID,
            CalendarContract.Calendars.CALENDAR_DISPLAY_NAME,
            CalendarContract.Calendars.CALENDAR_ACCESS_LEVEL,
            CalendarContract.Calendars.IS_PRIMARY,
            CalendarContract.Calendars.OWNER_ACCOUNT,
            CalendarContract.Calendars.ACCOUNT_NAME
        )
        val candidates = mutableListOf<WritableCalendarCandidate>()
        return try {
            context.contentResolver.query(
                CalendarContract.Calendars.CONTENT_URI,
                projection,
                "${CalendarContract.Calendars.VISIBLE}=1",
                null,
                null
            )?.use { cursor ->
                val idIndex = cursor.getColumnIndex(CalendarContract.Calendars._ID)
                val nameIndex = cursor.getColumnIndex(CalendarContract.Calendars.CALENDAR_DISPLAY_NAME)
                val accessIndex = cursor.getColumnIndex(CalendarContract.Calendars.CALENDAR_ACCESS_LEVEL)
                val primaryIndex = cursor.getColumnIndex(CalendarContract.Calendars.IS_PRIMARY)
                val ownerIndex = cursor.getColumnIndex(CalendarContract.Calendars.OWNER_ACCOUNT)
                val accountIndex = cursor.getColumnIndex(CalendarContract.Calendars.ACCOUNT_NAME)

                while (cursor.moveToNext()) {
                    val accessLevel = cursor.getIntOrDefault(accessIndex, 0)
                    if (accessLevel < CalendarContract.Calendars.CAL_ACCESS_CONTRIBUTOR) continue
                    candidates += WritableCalendarCandidate(
                        id = cursor.getLongOrDefault(idIndex, -1L),
                        displayName = cursor.getStringOrBlank(nameIndex).ifBlank { "Kalender" },
                        accessLevel = accessLevel,
                        isPrimary = cursor.getIntOrDefault(primaryIndex, 0) == 1,
                        isOwner = cursor.getStringOrBlank(ownerIndex).equals(
                            cursor.getStringOrBlank(accountIndex),
                            ignoreCase = true
                        )
                    )
                }
            }
            candidates
                .filter { it.id >= 0 }
                .sortedWith(
                    compareByDescending<WritableCalendarCandidate> { it.isPrimary }
                        .thenByDescending { it.isOwner }
                        .thenByDescending { it.accessLevel }
                )
                .firstOrNull()
        } catch (e: Exception) {
            Log.e(appTag, "Could not find writable calendar", e)
            null
        }
    }

    private fun calendarEventDescription(notes: String): String =
        notes.trim()

    private fun parseCalendarDateTimeMillis(raw: String, referenceDate: LocalDate = LocalDate.now(ZoneId.systemDefault())): Long? {
        val clean = raw.trim()
        if (clean.isBlank()) return null

        clean.toLongOrNull()
            ?.takeIf { it > 1_000_000_000_000L }
            ?.let { return it }

        runCatching { return ZonedDateTime.parse(clean).toInstant().toEpochMilli() }
        runCatching { return Instant.parse(clean).toEpochMilli() }

        calendarDateTimeFormatters().forEach { formatter ->
            runCatching {
                return LocalDateTime.parse(clean, formatter)
                    .atZone(ZoneId.systemDefault())
                    .toInstant()
                    .toEpochMilli()
            }
        }

        val date = calendarDateFromText(clean, referenceDate)
        val time = calendarTimeFromText(clean, date != null) ?: return null
        return LocalDateTime.of(date ?: referenceDate, time)
            .atZone(ZoneId.systemDefault())
            .toInstant()
            .toEpochMilli()
    }

    private fun calendarDateTimeFormatters(): List<DateTimeFormatter> =
        listOf(
            DateTimeFormatter.ISO_LOCAL_DATE_TIME,
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm", Locale.GERMANY),
            DateTimeFormatter.ofPattern("yyyy-MM-dd H:mm", Locale.GERMANY),
            DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm", Locale.GERMANY),
            DateTimeFormatter.ofPattern("dd.MM.yyyy H:mm", Locale.GERMANY),
            DateTimeFormatter.ofPattern("dd.MM.yy HH:mm", Locale.GERMANY),
            DateTimeFormatter.ofPattern("dd.MM.yy H:mm", Locale.GERMANY)
        )

    private fun calendarDateFromText(raw: String, referenceDate: LocalDate): LocalDate? {
        val text = semanticText(raw)
        if (text.contains("uebermorgen") || text.contains("ubermorgen")) return referenceDate.plusDays(2)
        if (text.contains("morgen") || text.contains("tomorrow")) return referenceDate.plusDays(1)
        if (text.contains("heute") || text.contains("today")) return referenceDate

        Regex("""\b(\d{1,2})\.(\d{1,2})(?:\.(\d{2,4}))?\b""").find(raw)?.let { match ->
            val day = match.groupValues.getOrNull(1)?.toIntOrNull() ?: return@let
            val month = match.groupValues.getOrNull(2)?.toIntOrNull() ?: return@let
            val rawYear = match.groupValues.getOrNull(3).orEmpty()
            val year = when {
                rawYear.length == 2 -> 2000 + rawYear.toInt()
                rawYear.length == 4 -> rawYear.toInt()
                else -> referenceDate.year
            }
            return runCatching { LocalDate.of(year, month, day) }.getOrNull()
        }

        val weekdays = mapOf(
            "montag" to DayOfWeek.MONDAY,
            "dienstag" to DayOfWeek.TUESDAY,
            "mittwoch" to DayOfWeek.WEDNESDAY,
            "donnerstag" to DayOfWeek.THURSDAY,
            "freitag" to DayOfWeek.FRIDAY,
            "samstag" to DayOfWeek.SATURDAY,
            "sonntag" to DayOfWeek.SUNDAY,
            "monday" to DayOfWeek.MONDAY,
            "tuesday" to DayOfWeek.TUESDAY,
            "wednesday" to DayOfWeek.WEDNESDAY,
            "thursday" to DayOfWeek.THURSDAY,
            "friday" to DayOfWeek.FRIDAY,
            "saturday" to DayOfWeek.SATURDAY,
            "sunday" to DayOfWeek.SUNDAY
        )
        weekdays.firstNotNullOfOrNull { (token, dayOfWeek) ->
            if (!text.contains(token)) null else dayOfWeek
        }?.let { dayOfWeek ->
            val delta = (dayOfWeek.value - referenceDate.dayOfWeek.value + 7) % 7
            return referenceDate.plusDays(delta.toLong())
        }

        return null
    }

    private fun calendarTimeFromText(raw: String, allowStandaloneHour: Boolean): LocalTime? {
        val lower = raw.lowercase(Locale.ROOT)
        Regex("""\b([01]?\d|2[0-3])[:.]([0-5]\d)\b""").find(lower)?.let { match ->
            val hour = match.groupValues[1].toInt()
            val minute = match.groupValues[2].toInt()
            return LocalTime.of(hour, minute)
        }

        Regex("""\b([01]?\d|2[0-3])\s*(?:uhr|h)\s*([0-5]\d)?\b""").find(lower)?.let { match ->
            val hour = match.groupValues[1].toInt()
            val minute = match.groupValues.getOrNull(2)?.takeIf { it.isNotBlank() }?.toIntOrNull() ?: 0
            return LocalTime.of(hour, minute)
        }

        Regex("""\b(?:um|at)\s+([01]?\d|2[0-3])\b""").find(lower)?.let { match ->
            return LocalTime.of(match.groupValues[1].toInt(), 0)
        }

        if (allowStandaloneHour) {
            Regex("""\b([01]?\d|2[0-3])\b""").findAll(lower).lastOrNull()?.let { match ->
                return LocalTime.of(match.groupValues[1].toInt(), 0)
            }
        }

        return null
    }

    private fun getAgenda(
        range: String,
        startEpochMillis: Long?,
        endEpochMillis: Long?,
        query: String
    ): String {
        if (!hasPermission(Manifest.permission.READ_CALENDAR)) {
            return "READ_CALENDAR-Berechtigung ist erforderlich, bevor ich die Agenda lesen kann."
        }

        val (startMillis, endMillis) = agendaRange(range, startEpochMillis, endEpochMillis)
        val builder = CalendarContract.Instances.CONTENT_URI.buildUpon()
        ContentUris.appendId(builder, startMillis)
        ContentUris.appendId(builder, endMillis)
        val projection = arrayOf(
            CalendarContract.Instances.EVENT_ID,
            CalendarContract.Instances.TITLE,
            CalendarContract.Instances.BEGIN,
            CalendarContract.Instances.END,
            CalendarContract.Instances.EVENT_LOCATION,
            CalendarContract.Instances.ALL_DAY,
            CalendarContract.Instances.CALENDAR_DISPLAY_NAME
        )
        val cleanQuery = query.trim().lowercase(Locale.ROOT)
        val events = mutableListOf<String>()

        return try {
            context.contentResolver.query(
                builder.build(),
                projection,
                null,
                null,
                "${CalendarContract.Instances.BEGIN} ASC"
            )?.use { cursor ->
                val titleIndex = cursor.getColumnIndex(CalendarContract.Instances.TITLE)
                val beginIndex = cursor.getColumnIndex(CalendarContract.Instances.BEGIN)
                val endIndex = cursor.getColumnIndex(CalendarContract.Instances.END)
                val locationIndex = cursor.getColumnIndex(CalendarContract.Instances.EVENT_LOCATION)
                val allDayIndex = cursor.getColumnIndex(CalendarContract.Instances.ALL_DAY)
                val calendarIndex = cursor.getColumnIndex(CalendarContract.Instances.CALENDAR_DISPLAY_NAME)

                while (cursor.moveToNext() && events.size < 12) {
                    val title = cursor.getStringOrBlank(titleIndex).ifBlank { "(ohne Titel)" }
                    val location = cursor.getStringOrBlank(locationIndex)
                    if (cleanQuery.isNotBlank() &&
                        !title.lowercase(Locale.ROOT).contains(cleanQuery) &&
                        !location.lowercase(Locale.ROOT).contains(cleanQuery)
                    ) {
                        continue
                    }
                    val begin = cursor.getLongOrDefault(beginIndex, startMillis)
                    val end = cursor.getLongOrDefault(endIndex, begin)
                    val allDay = cursor.getIntOrDefault(allDayIndex, 0) == 1
                    val calendarName = cursor.getStringOrBlank(calendarIndex)
                    val place = location.takeIf { it.isNotBlank() }?.let { " @ $it" }.orEmpty()
                    val calendarLabel = calendarName.takeIf { it.isNotBlank() }?.let { " [$it]" }.orEmpty()
                    events += "- ${formatAgendaTime(begin, end, allDay)}: $title$place$calendarLabel"
                }
            }

            if (events.isEmpty()) {
                "Keine Kalendertermine für ${formatDateRange(startMillis, endMillis)} gefunden."
            } else {
                "Agenda für ${formatDateRange(startMillis, endMillis)}:\n${events.joinToString("\n")}"
            }
        } catch (e: Exception) {
            Log.e(appTag, "Could not read agenda", e)
            "Konnte Agenda nicht lesen: ${e.message}"
        }
    }

    private fun findNearbyPlaces(category: String, radiusMeters: Int?, limit: Int?): String {
        val location = readLastKnownLocation()
            ?: return "Standort-Berechtigung oder ein aktueller GPS-Fix ist erforderlich, bevor ich nahe Orte finden kann."
        val cleanCategory = category.trim().ifBlank { "Orte" }
        val radius = (radiusMeters ?: 1000).coerceIn(100, 5000)
        val boundedLimit = (limit ?: NEARBY_RESULT_LIMIT).coerceIn(1, 12)
        val overpassError = StringBuilder()
        val overpassPlaces = runCatching {
            val query = buildOverpassNearbyQuery(cleanCategory, location.latitude, location.longitude, radius, boundedLimit)
            val raw = fetchOverpassText(query, "nahe Orte", maxChars = 250_000)
            if (raw.startsWith("Could not", ignoreCase = true) || raw.startsWith("Konnte", ignoreCase = true)) {
                overpassError.append(raw)
                emptyList()
            } else {
                parseNearbyPlaces(raw, cleanCategory, location.latitude, location.longitude)
            }
        }.getOrElse {
            Log.e(appTag, "Could not parse nearby places", it)
            overpassError.append("Konnte nahe Orte nicht auswerten: ${it.message}")
            emptyList()
        }

        val places = if (overpassPlaces.isNotEmpty()) {
            overpassPlaces
        } else {
            searchNominatimNearbyPlaces(cleanCategory, location.latitude, location.longitude, radius, boundedLimit)
        }
            .sortedBy { it.distanceMeters }
            .take(boundedLimit)

        if (places.isEmpty()) {
            val detail = overpassError.toString().takeIf { it.isNotBlank() }?.let { " ($it)" }.orEmpty()
            return "Keine nahen Orte für '$cleanCategory' im Umkreis von ${radius}m gefunden.$detail"
        }
        return places.mapIndexed { index, place ->
            "${index + 1}. ${place.name} (${formatDistance(place.distanceMeters.toDouble())}) - " +
                "${place.category}; ${formatCoordinates(place.latitude, place.longitude)}${place.detail.takeIf { it.isNotBlank() }?.let { "; $it" }.orEmpty()}"
        }.joinToString(prefix = "Nahe $cleanCategory:\n", separator = "\n")
    }

    private fun getTransitDepartures(stop: String, radiusMeters: Int?, limit: Int?): String {
        val boundedLimit = (limit ?: TRANSIT_RESULT_LIMIT).coerceIn(1, 12)
        val lookup = if (stop.isBlank()) {
            val location = readLastKnownLocation()
                ?: return "Standort-Berechtigung oder ein aktueller GPS-Fix ist erforderlich, bevor ich nahe Öffi-Haltestellen finden kann."
            val radius = (radiusMeters ?: 1200).coerceIn(100, 5000)
            searchOebbNearbyStops(location.latitude, location.longitude, radius)
        } else {
            searchOebbStopsByName(stop)
        }

        val selectedStop = lookup.stops.firstOrNull()
            ?: lookup.error?.let { return it }
            ?: return if (stop.isBlank()) "Keine nahen OEBB-Öffi-Haltestellen gefunden." else "Keine OEBB-Haltestelle für '$stop' gefunden."

        val scottyDepartures = fetchOebbScottyDepartures(selectedStop, boundedLimit)
        if (scottyDepartures.departures.isNotEmpty()) {
            val stopDistance = selectedStop.distanceMeters?.let { " (${formatDistance(it)})" }.orEmpty()
            return scottyDepartures.departures.joinToString(
                prefix = "Abfahrten von ${selectedStop.name}$stopDistance:\n",
                separator = "\n"
            )
        }

        val departuresRaw = fetchOebbText(
            "/stops/${Uri.encode(selectedStop.id)}/departures?duration=90&results=$boundedLimit",
            "OEBB-Abfahrten",
            maxChars = 120_000
        )
        if (departuresRaw.startsWith("Could not", ignoreCase = true) || departuresRaw.startsWith("Konnte", ignoreCase = true)) {
            return scottyDepartures.error ?: departuresRaw
        }

        val departures = runCatching { parseOebbDepartures(departuresRaw, boundedLimit) }.getOrElse {
            Log.e(appTag, "Could not parse OEBB departures", it)
            return scottyDepartures.error ?: "Konnte OEBB-Abfahrten nicht auswerten: ${it.message}"
        }
        if (departures.isEmpty()) return "Keine bevorstehenden Abfahrten für ${selectedStop.name} gefunden."

        val stopDistance = selectedStop.distanceMeters?.let { " (${formatDistance(it)})" }.orEmpty()
        return departures.joinToString(
            prefix = "Abfahrten von ${selectedStop.name}$stopDistance:\n",
            separator = "\n"
        )
    }

    private fun agendaRange(range: String, startEpochMillis: Long?, endEpochMillis: Long?): Pair<Long, Long> {
        if (startEpochMillis != null && endEpochMillis != null && endEpochMillis > startEpochMillis) {
            return startEpochMillis to endEpochMillis
        }

        val now = System.currentTimeMillis()
        val calendar = Calendar.getInstance(Locale.GERMANY)
        fun startOfDay(offsetDays: Int): Long {
            calendar.timeInMillis = now
            calendar.add(Calendar.DAY_OF_YEAR, offsetDays)
            calendar.set(Calendar.HOUR_OF_DAY, 0)
            calendar.set(Calendar.MINUTE, 0)
            calendar.set(Calendar.SECOND, 0)
            calendar.set(Calendar.MILLISECOND, 0)
            return calendar.timeInMillis
        }

        return when (range.trim().lowercase(Locale.ROOT)) {
            "tomorrow", "morgen" -> {
                val start = startOfDay(1)
                start to (start + 24 * 60 * 60 * 1000L - 1)
            }
            "next_24_hours", "naechste_24_stunden", "nächste_24_stunden", "24h", "next24" -> now to (now + 24 * 60 * 60 * 1000L)
            "week", "woche", "this_week" -> now to (now + 7 * 24 * 60 * 60 * 1000L)
            else -> {
                val start = startOfDay(0)
                start to (start + 24 * 60 * 60 * 1000L - 1)
            }
        }
    }

    private fun formatAgendaTime(begin: Long, end: Long, allDay: Boolean): String {
        if (allDay) return "ganztägig"
        val timeFormat = DateFormat.getTimeInstance(DateFormat.SHORT, Locale.GERMANY)
        return if (end > begin) {
            "${timeFormat.format(Date(begin))}-${timeFormat.format(Date(end))}"
        } else {
            timeFormat.format(Date(begin))
        }
    }

    private fun formatDateRange(startMillis: Long, endMillis: Long): String {
        val dateFormat = DateFormat.getDateInstance(DateFormat.SHORT, Locale.GERMANY)
        val startDate = dateFormat.format(Date(startMillis))
        val endDate = dateFormat.format(Date(endMillis))
        return if (startDate == endDate) startDate else "$startDate-$endDate"
    }

    private fun buildOverpassNearbyQuery(
        category: String,
        latitude: Double,
        longitude: Double,
        radiusMeters: Int,
        limit: Int
    ): String {
        val filters = osmFiltersForCategory(category)
        val body = filters.joinToString("\n") { filter ->
            """
                node(around:$radiusMeters,$latitude,$longitude)$filter;
                way(around:$radiusMeters,$latitude,$longitude)$filter;
            """.trimIndent()
        }
        val outputLimit = (limit * 3).coerceAtLeast(10)
        return """
            [out:json][timeout:6];
            (
            $body
            );
            out center tags $outputLimit;
        """.trimIndent()
    }

    private fun osmFiltersForCategory(category: String): List<String> {
        val normalized = semanticText(category)
        return when {
            normalized.contains("pharmacy") || normalized.contains("apotheke") ->
                listOf("[amenity=pharmacy]")
            normalized.contains("toilet") || normalized.contains("wc") || normalized.contains("restroom") || normalized.contains("klo") ->
                listOf("[amenity=toilets]")
            normalized.contains("atm") || normalized.contains("bankomat") || normalized.contains("geldautomat") ->
                listOf("[amenity=atm]")
            normalized.contains("supermarket") || normalized.contains("grocery") || normalized.contains("lebensmittel") || normalized.contains("supermarkt") ->
                listOf("[shop=supermarket]", "[shop=convenience]", "[shop=grocery]")
            normalized.contains("station") || normalized.contains("stop") || normalized.contains("haltestelle") || normalized.contains("bahnhof") ->
                listOf("[public_transport=station]", "[public_transport=platform]", "[railway=station]", "[railway=tram_stop]", "[highway=bus_stop]", "[amenity=bus_station]")
            normalized.contains("restaurant") || normalized.contains("essen") ->
                listOf("[amenity=restaurant]", "[amenity=fast_food]")
            normalized.contains("cafe") || normalized.contains("kaffee") ->
                listOf("[amenity=cafe]")
            normalized.contains("hospital") || normalized.contains("arzt") || normalized.contains("clinic") || normalized.contains("krankenhaus") ->
                listOf("[amenity=hospital]", "[amenity=clinic]", "[amenity=doctors]")
            normalized.contains("water") || normalized.contains("wasser") || normalized.contains("trink") ->
                listOf("[amenity=drinking_water]")
            normalized.contains("parking") || normalized.contains("parkplatz") || normalized.contains("parken") ->
                listOf("[amenity=parking]")
            normalized.contains("hotel") ->
                listOf("[tourism=hotel]", "[tourism=hostel]", "[tourism=guest_house]")
            else -> listOf("[name~\"${overpassRegexLiteral(category)}\",i]")
        }
    }

    private fun parseNearbyPlaces(rawJson: String, requestedCategory: String, originLat: Double, originLon: Double): List<NearbyPlace> {
        val elements = JSONObject(rawJson).optJSONArray("elements") ?: JSONArray()
        return buildList {
            for (index in 0 until elements.length()) {
                val element = elements.optJSONObject(index) ?: continue
                val center = element.optJSONObject("center")
                val lat = if (element.has("lat")) element.optDouble("lat") else center?.optDouble("lat") ?: Double.NaN
                val lon = if (element.has("lon")) element.optDouble("lon") else center?.optDouble("lon") ?: Double.NaN
                if (!lat.isFinite() || !lon.isFinite()) continue
                val tags = element.optJSONObject("tags") ?: JSONObject()
                val name = tags.optString("name")
                    .ifBlank { tags.optString("operator") }
                    .ifBlank { categoryFromTags(tags, requestedCategory) }
                val category = categoryFromTags(tags, requestedCategory)
                val distance = distanceMeters(originLat, originLon, lat, lon)
                val detail = nearbyDetail(tags)
                add(NearbyPlace(name, category, lat, lon, distance, detail))
            }
        }.distinctBy { "${it.name}:${it.latitude}:${it.longitude}" }
    }

    private fun searchNominatimNearbyPlaces(
        category: String,
        latitude: Double,
        longitude: Double,
        radiusMeters: Int,
        limit: Int
    ): List<NearbyPlace> {
        val url = nominatimNearbySearchUrl(category, latitude, longitude, radiusMeters, limit)
        val raw = fetchText(url, "nahe Orte per Nominatim", maxChars = 120_000)
        if (raw.startsWith("Could not", ignoreCase = true) || raw.startsWith("Konnte", ignoreCase = true)) {
            Log.w(appTag, raw)
            return emptyList()
        }

        return runCatching {
            parseNominatimNearbyPlaces(raw, category, latitude, longitude, radiusMeters)
        }.getOrElse {
            Log.e(appTag, "Could not parse nearby Nominatim places", it)
            emptyList()
        }
    }

    private fun nominatimNearbySearchUrl(
        category: String,
        latitude: Double,
        longitude: Double,
        radiusMeters: Int,
        limit: Int
    ): String {
        val latDelta = radiusMeters / 111_320.0
        val lonDelta = radiusMeters / (111_320.0 * cos(Math.toRadians(latitude)).coerceAtLeast(0.2))
        val left = longitude - lonDelta
        val right = longitude + lonDelta
        val top = latitude + latDelta
        val bottom = latitude - latDelta
        val query = URLEncoder.encode(nominatimNearbySearchTerm(category), "UTF-8")
        return "https://nominatim.openstreetmap.org/search" +
            "?format=jsonv2&limit=${(limit * 3).coerceAtLeast(10)}&countrycodes=at&accept-language=de" +
            "&addressdetails=1&extratags=1&dedupe=1&bounded=1" +
            "&viewbox=$left,$top,$right,$bottom&q=$query"
    }

    private fun nominatimNearbySearchTerm(category: String): String {
        val normalized = semanticText(category)
        return when {
            normalized.contains("pharmacy") || normalized.contains("apotheke") -> "Apotheke"
            normalized.contains("toilet") || normalized.contains("wc") || normalized.contains("restroom") || normalized.contains("klo") -> "Toiletten"
            normalized.contains("atm") || normalized.contains("bankomat") || normalized.contains("geldautomat") -> "Bankomat"
            normalized.contains("supermarket") || normalized.contains("grocery") || normalized.contains("lebensmittel") || normalized.contains("supermarkt") -> "Supermarkt"
            normalized.contains("station") || normalized.contains("stop") || normalized.contains("haltestelle") || normalized.contains("bahnhof") -> "Haltestelle"
            normalized.contains("restaurant") || normalized.contains("essen") -> "Restaurant"
            normalized.contains("cafe") || normalized.contains("kaffee") -> "Cafe"
            normalized.contains("hospital") || normalized.contains("arzt") || normalized.contains("clinic") || normalized.contains("krankenhaus") -> "Krankenhaus"
            normalized.contains("water") || normalized.contains("wasser") || normalized.contains("trink") -> "Trinkwasser"
            normalized.contains("parking") || normalized.contains("parkplatz") || normalized.contains("parken") -> "Parkplatz"
            normalized.contains("hotel") -> "Hotel"
            else -> category
        }
    }

    private fun parseNominatimNearbyPlaces(
        rawJson: String,
        requestedCategory: String,
        originLat: Double,
        originLon: Double,
        radiusMeters: Int
    ): List<NearbyPlace> {
        val array = JSONArray(rawJson)
        return buildList {
            for (index in 0 until array.length()) {
                val item = array.optJSONObject(index) ?: continue
                val lat = item.optString("lat").toDoubleOrNull() ?: continue
                val lon = item.optString("lon").toDoubleOrNull() ?: continue
                val distance = distanceMeters(originLat, originLon, lat, lon)
                if (distance > radiusMeters * 1.5) continue
                val displayName = item.optString("display_name")
                val name = item.optString("name")
                    .ifBlank { displayName.split(',').firstOrNull()?.trim().orEmpty() }
                    .ifBlank { requestedCategory }
                val category = item.optString("type")
                    .ifBlank { item.optString("class") }
                    .ifBlank { requestedCategory }
                val detail = displayName
                    .split(',')
                    .drop(1)
                    .map { it.trim() }
                    .filter { it.isNotBlank() }
                    .take(2)
                    .joinToString(", ")
                add(NearbyPlace(name, category, lat, lon, distance, detail))
            }
        }.distinctBy { "${semanticText(it.name)}:${"%.5f".format(Locale.US, it.latitude)}:${"%.5f".format(Locale.US, it.longitude)}" }
    }

    private fun categoryFromTags(tags: JSONObject, fallback: String): String =
        listOf("amenity", "shop", "tourism", "railway", "public_transport", "highway")
            .firstNotNullOfOrNull { key -> tags.optString(key).takeIf { it.isNotBlank() } }
            ?: fallback

    private fun nearbyDetail(tags: JSONObject): String {
        val address = listOf(
            tags.optString("addr:street"),
            tags.optString("addr:housenumber"),
            tags.optString("addr:postcode"),
            tags.optString("addr:city")
        ).filter { it.isNotBlank() }.joinToString(" ")
        val opening = tags.optString("opening_hours").takeIf { it.isNotBlank() }?.let { "open: $it" }
        return listOf(address.takeIf { it.isNotBlank() }, opening).filterNotNull().joinToString("; ")
    }

    private fun searchOebbNearbyStops(latitude: Double, longitude: Double, radiusMeters: Int): TransitStopLookup {
        val body = fetchOebbText(
            "/stops/nearby?latitude=$latitude&longitude=$longitude&distance=$radiusMeters&results=5",
            "nahe OEBB-Haltestellen",
            maxChars = 120_000
        )
        if (body.startsWith("Could not", ignoreCase = true) || body.startsWith("Konnte", ignoreCase = true)) {
            Log.w(appTag, body)
            return TransitStopLookup(error = body)
        }
        return runCatching {
            TransitStopLookup(stops = parseOebbStops(body))
        }.getOrElse {
            Log.e(appTag, "Could not parse nearby OEBB stops", it)
            TransitStopLookup(error = "Konnte nahe OEBB-Haltestellen nicht auswerten: ${it.message}")
        }
    }

    private fun searchOebbStopsByName(stop: String): TransitStopLookup {
        val scottyLookup = searchOebbScottyStopsByName(stop)
        if (scottyLookup.stops.isNotEmpty()) return scottyLookup

        val body = fetchOebbText(
            "/locations?query=${URLEncoder.encode(stop, "UTF-8")}&results=5&stops=true&addresses=false&poi=false",
            "OEBB-Haltestellensuche",
            maxChars = 120_000
        )
        if (body.startsWith("Could not", ignoreCase = true) || body.startsWith("Konnte", ignoreCase = true)) {
            Log.w(appTag, body)
            return if (scottyLookup.error != null) {
                TransitStopLookup(error = "${scottyLookup.error}; $body")
            } else {
                TransitStopLookup(error = body)
            }
        }
        return runCatching {
            TransitStopLookup(stops = parseOebbStops(body))
        }.getOrElse {
            Log.e(appTag, "Could not parse OEBB stop search", it)
            TransitStopLookup(error = scottyLookup.error ?: "Konnte OEBB-Haltestellensuche nicht auswerten: ${it.message}")
        }
    }

    private fun searchOebbScottyStopsByName(stop: String): TransitStopLookup {
        val cleanStop = stop.trim()
        if (cleanStop.isBlank()) return TransitStopLookup()
        val url = "$OEBB_SCOTTY_BASE_URL/bin/ajax-getstop.exe/dn?REQ0JourneyStopsS0A=1&S=${URLEncoder.encode(cleanStop, "UTF-8")}&js=true&"
        val body = fetchText(url, "OEBB-Scotty-Haltestellensuche", maxChars = 120_000)
        if (body.startsWith("Could not", ignoreCase = true) || body.startsWith("Konnte", ignoreCase = true)) {
            Log.w(appTag, body)
            return TransitStopLookup(error = body)
        }
        return runCatching {
            val obj = JSONObject(extractJavascriptObject(body, "SLs.sls"))
            val suggestions = obj.optJSONArray("suggestions") ?: JSONArray()
            val stops = buildList {
                for (index in 0 until suggestions.length()) {
                    val suggestion = suggestions.optJSONObject(index) ?: continue
                    val name = decodeHtmlEntities(suggestion.optString("value")).trim()
                    val extId = suggestion.optString("extId")
                        .ifBlank { Regex("@L=([^@]+)").find(suggestion.optString("id"))?.groupValues?.getOrNull(1).orEmpty() }
                    val id = normalizeOebbEvaId(extId)
                    if (name.isBlank() || id.isBlank()) continue
                    add(TransitStop(id = id, name = name))
                }
            }
            TransitStopLookup(stops = stops)
        }.getOrElse {
            Log.e(appTag, "Could not parse OEBB Scotty stop search", it)
            TransitStopLookup(error = "Konnte OEBB-Scotty-Haltestellensuche nicht auswerten: ${it.message}")
        }
    }

    private fun parseOebbStops(rawJson: String): List<TransitStop> {
        val array = jsonArrayFromResponse(rawJson, "stops", "locations")
        return buildList {
            for (index in 0 until array.length()) {
                val stop = array.optJSONObject(index) ?: continue
                val type = stop.optString("type")
                if (type.isNotBlank() && type !in setOf("stop", "station", "location")) continue
                val id = stop.optString("id").ifBlank { stop.optString("stationId") }
                val name = stop.optString("name")
                if (id.isBlank() || name.isBlank()) continue
                val distance = if (stop.has("distance") && !stop.isNull("distance")) stop.optDouble("distance") else null
                add(TransitStop(id, name, distance))
            }
        }
    }

    private fun fetchOebbScottyDepartures(stop: TransitStop, limit: Int): TransitDeparturesLookup {
        val evaId = normalizeOebbEvaId(stop.id)
        if (evaId.isBlank()) return TransitDeparturesLookup(error = "Keine gültige OEBB-Haltestellen-ID für ${stop.name}.")
        val url = "$OEBB_SCOTTY_BASE_URL/bin/stboard.exe/dn?" +
            "L=vs_liveticker&evaId=${Uri.encode(evaId)}&dirInput=&boardType=dep&selectDate=today&time=now&" +
            "productsFilter=1111111111&additionalTime=0&useEquiv=no&outputMode=tickerDataOnly&maxJourneys=${limit.coerceAtLeast(1)}&start=yes"
        val body = fetchText(url, "OEBB-Scotty-Abfahrten", maxChars = 120_000)
        if (body.startsWith("Could not", ignoreCase = true) || body.startsWith("Konnte", ignoreCase = true)) {
            Log.w(appTag, body)
            return TransitDeparturesLookup(error = body)
        }
        return runCatching {
            val obj = JSONObject(extractJavascriptObject(body, "journeysObj"))
            val journeys = obj.optJSONArray("journey") ?: JSONArray()
            val departures = buildList {
                for (index in 0 until journeys.length()) {
                    if (size >= limit) break
                    val journey = journeys.optJSONObject(index) ?: continue
                    val time = journey.optString("rti")
                        .ifBlank { journey.optString("ti") }
                        .ifBlank { "bald" }
                    val line = decodeHtmlEntities(journey.optString("pr")).trim().ifBlank { "Linie" }
                    val direction = decodeHtmlEntities(
                        journey.optString("st")
                            .ifBlank { journey.optString("lastStop") }
                    ).trim().ifBlank { "unbekannt" }
                    val plannedTime = journey.optString("ti")
                    val delay = if (time != plannedTime && plannedTime.isNotBlank()) " statt $plannedTime" else ""
                    val platform = decodeHtmlEntities(journey.optString("tr"))
                        .trim()
                        .takeIf { it.isNotBlank() }
                        ?.let { " Gleis/Steig $it" }
                        .orEmpty()
                    add("- $time$delay: $line nach $direction$platform")
                }
            }
            TransitDeparturesLookup(departures = departures)
        }.getOrElse {
            Log.e(appTag, "Could not parse OEBB Scotty departures", it)
            TransitDeparturesLookup(error = "Konnte OEBB-Scotty-Abfahrten nicht auswerten: ${it.message}")
        }
    }

    private fun parseOebbDepartures(rawJson: String, limit: Int): List<String> {
        val array = jsonArrayFromResponse(rawJson, "departures", "results")
        return buildList {
            for (index in 0 until array.length()) {
                if (size >= limit) break
                val departure = array.optJSONObject(index) ?: continue
                val line = departure.optJSONObject("line")
                val lineName = line?.optString("name")?.ifBlank { line.optString("fahrtNr") } ?: "Linie"
                val direction = departure.optString("direction").ifBlank { departure.optString("destination") }
                val whenIso = departure.optString("when").ifBlank { departure.optString("plannedWhen") }
                val time = formatIsoTime(whenIso).ifBlank { "bald" }
                val delaySeconds = if (departure.has("delay") && !departure.isNull("delay")) departure.optInt("delay") else 0
                val delay = if (delaySeconds > 0) " +${delaySeconds / 60}min" else ""
                val platform = departure.optString("platform")
                    .ifBlank { departure.optString("plannedPlatform") }
                    .takeIf { it.isNotBlank() }
                    ?.let { " Gleis/Steig $it" }
                    .orEmpty()
                add("- $time$delay: $lineName nach ${direction.ifBlank { "unbekannt" }}$platform")
            }
        }
    }

    private fun jsonArrayFromResponse(rawJson: String, vararg fieldNames: String): JSONArray {
        val trimmed = rawJson.trim()
        if (trimmed.startsWith("[")) return JSONArray(trimmed)
        val obj = JSONObject(trimmed)
        fieldNames.forEach { name ->
            obj.optJSONArray(name)?.let { return it }
        }
        return JSONArray()
    }

    private fun extractJavascriptObject(raw: String, assignmentName: String): String {
        val assignmentIndex = raw.indexOf(assignmentName)
        if (assignmentIndex < 0) throw IllegalArgumentException("$assignmentName nicht gefunden")
        val equalsIndex = raw.indexOf('=', assignmentIndex)
        if (equalsIndex < 0) throw IllegalArgumentException("$assignmentName ohne Zuweisung")
        val start = raw.indexOf('{', equalsIndex)
        if (start < 0) throw IllegalArgumentException("$assignmentName ohne Objekt")
        var depth = 0
        var inString = false
        var escaping = false
        for (index in start until raw.length) {
            val ch = raw[index]
            if (inString) {
                when {
                    escaping -> escaping = false
                    ch == '\\' -> escaping = true
                    ch == '"' -> inString = false
                }
            } else {
                when (ch) {
                    '"' -> inString = true
                    '{' -> depth++
                    '}' -> {
                        depth--
                        if (depth == 0) return raw.substring(start, index + 1)
                    }
                }
            }
        }
        throw IllegalArgumentException("$assignmentName unvollständig")
    }

    private fun normalizeOebbEvaId(value: String): String =
        value.trim().trimStart('0').ifBlank { value.trim() }

    private fun fetchOebbText(path: String, label: String, maxChars: Int): String {
        val api = fetchText("$OEBB_API_BASE_URL/api$path", label, maxChars)
        if (!api.contains("HTTP 404", ignoreCase = true)) return api
        return fetchText("$OEBB_API_BASE_URL$path", label, maxChars)
    }

    private fun fetchOverpassText(bodyText: String, label: String, maxChars: Int): String {
        var lastError = "Konnte $label nicht abrufen."
        OVERPASS_API_URLS.forEach { url ->
            repeat(2) { attempt ->
                val result = fetchTextPostDetailed(url, bodyText, label, maxChars)
                result.text?.let { return it }
                lastError = result.errorMessage ?: lastError
                if (!isTransientOverpassStatus(result.httpCode)) return lastError
                runCatching { Thread.sleep(250L * (attempt + 1)) }
            }
        }
        return if (lastError.contains("HTTP 409") || lastError.contains("HTTP 429")) {
            "$lastError Overpass ist gerade ausgelastet; bitte gleich noch einmal versuchen."
        } else {
            lastError
        }
    }

    private fun isTransientOverpassStatus(httpCode: Int?): Boolean =
        httpCode in setOf(409, 429, 502, 503, 504)

    private fun fetchTextPost(url: String, bodyText: String, label: String, maxChars: Int = Int.MAX_VALUE): String {
        val result = fetchTextPostDetailed(url, bodyText, label, maxChars)
        return result.text ?: result.errorMessage ?: "Konnte $label nicht abrufen."
    }

    private fun fetchTextPostDetailed(url: String, bodyText: String, label: String, maxChars: Int = Int.MAX_VALUE): TextFetchResult = try {
        val request = Request.Builder()
            .url(url)
            .post(bodyText.toRequestBody("text/plain; charset=utf-8".toMediaType()))
            .header("User-Agent", WEB_SEARCH_USER_AGENT)
            .header("Accept-Language", "de,en;q=0.8")
            .build()
        getWebFetchClient().newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                TextFetchResult(httpCode = response.code, errorMessage = "Konnte $label nicht abrufen: HTTP ${response.code}")
            } else {
                val text = response.body?.string()?.trim().orEmpty()
                TextFetchResult(text = if (maxChars == Int.MAX_VALUE) text else text.take(maxChars), httpCode = response.code)
            }
        }
    } catch (e: Exception) {
        TextFetchResult(errorMessage = "Konnte $label nicht abrufen: ${e.message}")
    }

    private fun explicitCoordinates(latitudeRaw: String, longitudeRaw: String): LocationCoordinates? {
        val latitude = latitudeRaw.toDoubleOrNull()
        val longitude = longitudeRaw.toDoubleOrNull()
        if (latitude == null || longitude == null) return null
        if (latitude !in -90.0..90.0 || longitude !in -180.0..180.0) return null
        return LocationCoordinates(latitude, longitude, 0f)
    }

    private fun resolvedPlaceCoordinates(address: String): LocationCoordinates? {
        val cleanAddress = address.trim()
        if (cleanAddress.isBlank()) return null
        return runCatching {
            navigationDestinationResolver.resolve(cleanAddress, "foot", maxCandidates = 1).selected?.let {
                LocationCoordinates(it.latitude, it.longitude, 0f)
            }
        }.getOrNull()
    }

    private fun defaultSavedPlaceName(
        description: String,
        address: String,
        coordinates: LocationCoordinates,
        now: Long
    ): String {
        description.trim().takeIf { it.isNotBlank() }?.let { return it.take(48) }
        address.trim().takeIf { it.isNotBlank() }?.let { return it.take(48) }
        val timestamp = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT, Locale.GERMANY).format(Date(now))
        return "Ort $timestamp (${formatCoordinates(coordinates.latitude, coordinates.longitude)})"
    }

    private fun placeDetail(place: SavedPlaceRecord): String =
        listOf(place.description, place.address)
            .filter { it.isNotBlank() }
            .joinToString(" - ")
            .takeIf { it.isNotBlank() }
            ?.let { "; $it" }
            .orEmpty()

    private fun parseStringList(raw: String): List<String> {
        val clean = raw.trim()
        if (clean.isBlank()) return emptyList()
        if (clean.startsWith("[")) {
            return runCatching {
                val array = JSONArray(clean)
                buildList {
                    for (index in 0 until array.length()) {
                        array.optString(index).trim().takeIf { it.isNotBlank() }?.let { add(it.take(40)) }
                    }
                }
            }.getOrDefault(emptyList())
        }
        return clean.split(',', ';', '#')
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .map { it.take(40) }
    }

    private fun inferMemoryCategory(content: String): String =
        if (looksLikeIdentityMemory(content)) "identitaet" else "general"

    private fun inferredMemoryImportance(content: String, category: String): Int =
        if (isIdentityMemory(content, category, emptyList())) 5 else 1

    private fun enrichedMemoryTags(tags: List<String>, content: String, category: String): List<String> {
        val inferred = if (isIdentityMemory(content, category, tags)) listOf("identitaet", "name") else emptyList()
        return (tags + inferred)
            .map { it.trim().take(40) }
            .filter { it.isNotBlank() }
            .distinctBy { semanticText(it) }
    }

    private fun passiveMemoryQuery(instruction: String): String {
        val additions = mutableListOf<String>()
        if (looksLikeIdentityMemoryQuestion(instruction)) additions += "name identitaet identity called heisse"
        return (listOf(instruction) + additions).joinToString(" ")
    }

    private fun looksLikeIdentityMemoryQuestion(value: String): Boolean {
        val text = semanticText(value)
        return listOf(
            "wie heisse ich",
            "was ist mein name",
            "wie ist mein name",
            "wer bin ich",
            "what is my name",
            "who am i"
        ).any { text.contains(it) }
    }

    private fun looksLikeIdentityMemory(value: String): Boolean {
        val text = semanticText(value)
        return listOf(
            "mein name ist",
            "ich heisse",
            "ich bin",
            "my name is",
            "i am called"
        ).any { text.contains(it) }
    }

    private fun isIdentityMemory(content: String, category: String, tags: List<String>): Boolean {
        val categoryText = semanticText(category)
        val tagText = semanticText(tags.joinToString(" "))
        val contentText = semanticText(content)
        val contentTokens = semanticTokens(content)
        return categoryText in setOf("identitaet", "identitat", "identity", "person", "profil", "profile") ||
            "identitaet" in semanticTokens(tagText) ||
            "identitat" in semanticTokens(tagText) ||
            "identity" in semanticTokens(tagText) ||
            looksLikeIdentityMemory(content) ||
            ("name" in contentTokens && (contentText.contains("mein") || contentText.startsWith("name ")))
    }

    private fun semanticOverlapScore(query: String, content: String, tags: List<String>, category: String): Double {
        val queryTokens = semanticTokens(query)
        if (queryTokens.isEmpty()) return 0.0
        val contentTokens = semanticTokens("$content ${tags.joinToString(" ")} $category")
        if (contentTokens.isEmpty()) return 0.0
        val overlap = queryTokens.count { it in contentTokens }
        val phraseBonus = if (semanticText(content).contains(semanticText(query))) 4.0 else 0.0
        val tagBonus = tags.count { tag -> semanticTokens(tag).any { it in queryTokens } } * 1.5
        return overlap * 2.0 + phraseBonus + tagBonus
    }

    private fun semanticTokens(value: String): Set<String> =
        semanticText(value)
            .split(' ')
            .asSequence()
            .map { it.trim() }
            .filter { it.length >= 3 && it !in SEMANTIC_STOP_WORDS }
            .flatMap { semanticTokenAliases(it).asSequence() }
            .toSet()

    private fun semanticTokenAliases(token: String): Set<String> =
        when (token) {
            "heisse", "heisst", "heissen", "genannt", "called" -> setOf(token, "name")
            "name", "namen" -> setOf(token, "heisse")
            "identitaet", "identitat", "identity", "profil", "profile" -> setOf(token, "identitaet", "identitat", "identity")
            else -> setOf(token)
        }

    private fun semanticText(value: String): String {
        val normalized = Normalizer.normalize(value.lowercase(Locale.ROOT), Normalizer.Form.NFD)
            .replace(Regex("\\p{Mn}+"), "")
            .replace("ß", "ss")
        return normalized.map { char ->
            if (char.isLetterOrDigit()) char else ' '
        }.joinToString("").collapseWhitespace()
    }

    private fun shortId(): String =
        UUID.randomUUID().toString().substring(0, 8)

    private fun formatCoordinates(latitude: Double, longitude: Double): String =
        "%.5f, %.5f".format(Locale.US, latitude, longitude)

    private fun formatDistance(meters: Double): String =
        if (meters >= 1000.0) {
            "%.1f km".format(Locale.GERMANY, meters / 1000.0)
        } else {
            "${meters.toInt()} m"
        }

    private fun distanceMeters(fromLat: Double, fromLon: Double, toLat: Double, toLon: Double): Float {
        val result = FloatArray(1)
        android.location.Location.distanceBetween(fromLat, fromLon, toLat, toLon, result)
        return result[0]
    }

    private fun overpassRegexLiteral(value: String): String =
        value.trim().take(80).replace("\\", "\\\\").replace("\"", "\\\"")

    private fun formatIsoTime(iso: String): String =
        runCatching {
            ZonedDateTime.parse(iso).format(DateTimeFormatter.ofPattern("HH:mm", Locale.GERMANY))
        }.getOrDefault("")

    private fun android.database.Cursor.getStringOrBlank(index: Int): String =
        if (index >= 0 && !isNull(index)) getString(index).orEmpty() else ""

    private fun android.database.Cursor.getLongOrDefault(index: Int, defaultValue: Long): Long =
        if (index >= 0 && !isNull(index)) getLong(index) else defaultValue

    private fun android.database.Cursor.getIntOrDefault(index: Int, defaultValue: Int): Int =
        if (index >= 0 && !isNull(index)) getInt(index) else defaultValue

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
        val memoryContext = semanticMemoryContext(instruction)
        val sessionPrompt = when {
            hasImage ->
                "Diese Anfrage enthält ein frisches Kamerabild. Verwende dieses Bild als primäre Quelle und ziehe keine älteren Bildbeschreibungen oder Werkzeug-Ergebnisse heran."
            requestChatId != null ->
                "Du bist in einem explizit gewählten persistenten Chat. Verwende dessen Verlauf. Nutze Chat-Verwaltungswerkzeuge nur, wenn der Nutzer neue Chats, Chatlisten, Umbenennungen, Löschungen, Wechsel oder das Verlassen des persistenten Chats wünscht."
            conversationHistory.isNotEmpty() ->
                "Du bist in der ephemeren Hauptkonversation. Du hast nur den Verlauf dieser gerade aktiven Brillen-Konversation. Erstelle oder betrete persistente Chats nur bei ausdrücklicher Nutzerbitte."
            else ->
                "Diese Anfrage ist Teil der ephemeren Hauptkonversation und hat keinen gespeicherten Chat-Verlauf. Erstelle oder betrete persistente Chats nur bei ausdrücklicher Nutzerbitte."
        }
        val systemPrompt = getSystemPrompt() +
            "\n\nDu heißt Neuro. Deutsch (Österreich), kurz und freihändig. ${currentAssistantTimeHint()} " +
            "Nutze Werkzeuge nur bei klarer Handlung, nie für kurze Tests wie hallo/test/ping. Sichtfragen brauchen brillenfoto_aufnehmen. " +
            "Navigation: nutze navigation mit action=start, confirm oder stop; target ist das Ziel, selection die Nutzerwahl; gespeicherte Ortsnamen wie Zuhause/Home/Parkplatz werden lokal bevorzugt. oeffi ist Standard, zu_fuss nur bei Fußweg. Vertraue dem besten Zieltreffer, außer die Werkzeug-Antwort fragt ausdrücklich nach Auswahl. " +
            "Öffne keine externe Navi-App. Kontaktname reicht für Anruf/SMS. Timer/Erinnerungen/Kalender: berechne exakte seconds oder epoch_millis selbst und übergib numerische Zeitwerte als Dezimalziffern-Strings; Kalender-Start/Ende gehören nie in notes. " +
            "Semantische Erinnerung: gedaechtnis action=save nur für ausdrückliches Merken oder stabile persönliche Fakten; search/list/delete für Erinnerungsfragen oder Löschwunsch. " +
            "Gespeicherte Orte: orte action=save/list/navigate/delete für Parkplatz, Hotel, Eingang, Treffpunkt und ähnliche Orte; save darf ohne name den aktuellen GPS-Ort speichern; nutze delete statt vagem 'vergessen'. " +
            "Kalender: agenda_abrufen für Termin-/Agenda-Abfragen. Umgebung: nahe_orte_finden für nahe Orte; abfahrten_abrufen für Bus/Bahn/Bim/U-Bahn-Abfahrten. " +
            "QR-/Barcode-Fragen sind Sichtfragen und brauchen brillenfoto_aufnehmen; lies sichtbaren Code/URL/Text aus dem frischen Bild, soweit erkennbar. " +
            "Rokid-Brillenapps: rokid_app_oeffnen, auch für uebersetzer. Aktuelles/Web/Preise/Öffnungszeiten: web_suche und aus den zurückgegebenen Webfunden antworten, keinen Browser öffnen. Persistente Chats: chat action=new/list/switch/rename/leave/delete nur bei ausdrücklichem Wunsch. $sessionPrompt" +
            memoryContext.takeIf { it.isNotBlank() }?.let { "\n\n$it" }.orEmpty()
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
                    content = "Gib jetzt eine knappe deutschsprachige Erfolgsmeldung oder Fehlermeldung zu diesen Werkzeug-Ergebnissen aus. Frage nicht erneut nach bereits aufgelösten Daten:\n${toolResolution.results.joinToString("\n")}"
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
            val toolName = internalToolName(it.substringBefore(":").trim())
            toolName == "leave_chat" ||
                toolName == "delete_chat" ||
                (toolName == "chat" && (
                    it.contains("Persistenten Chat verlassen", ignoreCase = true) ||
                        it.contains("Persistenten Chat gelöscht", ignoreCase = true)
                    ))
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
                        val toolName = internalToolName(toolCall.function.name)
                        val result = executeAssistantTool(toolCall, instruction)
                        if (result == DEFERRED_PHOTO_TOOL_RESULT || result == LOCAL_TOOL_HANDLED_RESULT) {
                            Log.d(appTag, "Tool $toolName handled locally; deferring final assistant response")
                            return ToolResolutionResult(messagesList, toolResults, deferred = true)
                        }
                        toolResults += "${exposedToolName(toolName)}: $result"
                        listener?.onAssistantToolResult(toolName, result)
                        if (shouldDeferAfterToolResult(toolName, result)) {
                            Log.d(appTag, "Tool $toolName completed locally; deferring final assistant response")
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

                    if (toolCalls.any { internalToolName(it.function.name) == "stop_conversation" }) return ToolResolutionResult(messagesList, toolResults)
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
