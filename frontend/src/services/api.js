const API_BASE = "http://localhost:8000";

export async function sendMessage(message) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: "default_user",
      message,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));

    throw new Error(
      error.detail || "Chat request failed"
    );
  }

  return response.json();
}

export async function submitFeedback(
  messageId,
  feedback
) {
  const response = await fetch(
    `${API_BASE}/api/feedback`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message_id: messageId,
        feedback,
      }),
    }
  );

  if (!response.ok) {
    throw new Error("Feedback request failed");
  }

  return response.json();
}

/**
 * Streams the chat response token by token using Server-Sent Events (SSE).
 * @param {string} message - The user prompt
 * @param {function} onToken - Callback fired when a new token chunk arrives
 * @param {function} onComplete - Callback fired when stream finishes, returning metadata (message_id, scores)
 */
export async function streamMessage(message, onToken, onComplete) {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      user_id: "default_user",
      message,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Chat stream request failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let accumulatedText = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    const lines = chunk.split("\n\n");

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.replace("data: ", ""));
          
          if (data.done) {
            if (onComplete) onComplete(data);
          } else if (data.token) {
            accumulatedText += data.token;
            if (onToken) onToken(accumulatedText);
          }
        } catch (e) {
          console.error("Failed to parse SSE data chunk:", e);
        }
      }
    }
  }
}