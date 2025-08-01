export interface PredictionRequest {
  // Use the same structure you send to SageMaker. For now, accept any shape.
  [key: string]: any;
}

export interface PredictionResponse {
  // Adjust according to the model's response schema
  [key: string]: any;
}

/**
 * callPredictAPI
 * Sends a POST request to /api/predict with the provided payload and returns the parsed JSON response.
 */
export async function callPredictAPI(
  payload: PredictionRequest
): Promise<PredictionResponse> {
  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(
      `Prediction API error (status ${res.status}): ${errorBody.error ?? res.statusText}`
    );
  }

  return (await res.json()) as PredictionResponse;
}
