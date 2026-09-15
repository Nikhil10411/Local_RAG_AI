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