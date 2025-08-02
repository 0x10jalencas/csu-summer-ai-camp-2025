import { NextRequest, NextResponse } from 'next/server';
import { InvokeEndpointCommand } from '@aws-sdk/client-sagemaker-runtime';
import { client } from '@/lib/predict';

/**
 * POST /api/predict
 * Invokes an AWS SageMaker endpoint with the request payload and returns the prediction.
 *
 * Environment variables required:
 * - AWS_ACCESS_KEY_ID
 * - AWS_SECRET_ACCESS_KEY
 * - AWS_REGION
 * - SAGEMAKER_ENDPOINT_NAME
 */
export async function POST(req: NextRequest) {
  console.log('[API DEBUG] ============ API ROUTE STARTED ============');
  
  try {
    console.log('[API DEBUG] Request method:', req.method);
    console.log('[API DEBUG] Request URL:', req.url);
    console.log('[API DEBUG] Request headers:', Object.fromEntries(req.headers.entries()));
    
    // Parse incoming JSON payload from the request body
    console.log('[API DEBUG] Attempting to parse request body...');
    let payload;
    try {
      payload = await req.json();
      console.log('[API DEBUG] Successfully parsed request body');
    } catch (parseError) {
      console.error('[API ERROR] Failed to parse request body as JSON:', parseError);
      return NextResponse.json({
        error: 'Invalid JSON in request body',
        debugError: {
          message: 'Failed to parse request body as JSON',
          parseError: parseError instanceof Error ? parseError.message : String(parseError)
        }
      }, { status: 400 });
    }
    
    console.log('[API DEBUG] ============ PREDICTION REQUEST DEBUG ============');
    console.log('[API DEBUG] Received payload:', JSON.stringify(payload, null, 2));
    console.log('[API DEBUG] Payload type:', typeof payload);
    console.log('[API DEBUG] Payload keys:', Object.keys(payload));
    
    if (payload.features) {
      console.log('[API DEBUG] Features array found!');
      console.log('[API DEBUG] Features array length:', payload.features.length);
      console.log('[API DEBUG] Features array:', payload.features);
      console.log('[API DEBUG] Features types:', payload.features.map((f: unknown) => typeof f));
      console.log('[API DEBUG] All features are numbers?', payload.features.every((f: unknown) => typeof f === 'number'));
    } else {
      console.log('[API DEBUG] NO FEATURES KEY FOUND IN PAYLOAD!');
    }

    const region = process.env.AWS_REGION;
    const endpointName = process.env.SAGEMAKER_ENDPOINT_NAME ?? 'student-risk-22f-sigmoid-endpoint-1754044496';
    const mockMode = process.env.MOCK_PREDICTION;
    const hasAwsCredentials = !!(process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY);

    console.log('[API DEBUG] ============ ENVIRONMENT CHECK ============');
    console.log('[API DEBUG] AWS Region:', region);
    console.log('[API DEBUG] Endpoint Name:', endpointName);
    console.log('[API DEBUG] Mock mode?:', mockMode);
    console.log('[API DEBUG] Has AWS credentials?:', hasAwsCredentials);
    console.log('[API DEBUG] AWS_ACCESS_KEY_ID exists?:', !!process.env.AWS_ACCESS_KEY_ID);
    console.log('[API DEBUG] AWS_SECRET_ACCESS_KEY exists?:', !!process.env.AWS_SECRET_ACCESS_KEY);

    if (!region) {
      console.error('[API ERROR] Missing AWS_REGION environment variable');
      return NextResponse.json(
        {
          error: 'AWS_REGION must be configured in environment variables.',
          debug: {
            region: region,
            mockMode: mockMode,
            hasCredentials: hasAwsCredentials
          }
        },
        { status: 500 }
      );
    }

    // If MOCK_PREDICTION env var is set, skip the real invocation and return a mock
    if (process.env.MOCK_PREDICTION === 'true') {
      console.log('[API DEBUG] Returning mock prediction because MOCK_PREDICTION=true');
      return NextResponse.json({ 
        prediction: 0.75, 
        received: payload,
        debug: {
          message: 'This is a mock response',
          featuresReceived: payload.features?.length || 'no features key',
          mockMode: true
        }
      });
    }

    // === SENDING TO SAGEMAKER ===

    // === Build request body =================================================
    // Our Sigmoid‑Neuron model now expects **all 22 encoded features** in the
    // exact order produced by the shared encoder.
    const originalFeatures: unknown[] = Array.isArray(payload.features)
      ? payload.features
      : [];

    console.log('[API DEBUG] Incoming features length:', originalFeatures.length);
    if (originalFeatures.length !== 22) {
      return NextResponse.json(
        {
          error: `Model expects 22 features, received ${originalFeatures.length}.`,
          receivedLength: originalFeatures.length,
        },
        { status: 400 }
      );
    }

    // Sagemaker inference container accepts the raw array as JSON.
    const requestData = originalFeatures;

    try {
      const command = new InvokeEndpointCommand({
        EndpointName: endpointName,
        Body: Buffer.from(JSON.stringify(requestData)),
        ContentType: 'application/json',
        Accept: 'application/json',
      });

      console.log('[API DEBUG] Sending request to SageMaker...');
      const response = await client.send(command);

      // Convert the binary body back to string
      const responseBody = new TextDecoder('utf-8').decode(response.Body as Uint8Array);
      console.log('[API DEBUG] SUCCESS! Raw response body:', responseBody);

      const parsedResponse = JSON.parse(responseBody);
      console.log('[API DEBUG] SUCCESS! Parsed response:', parsedResponse);
      
      // Success! Return the result with metadata
      return NextResponse.json({
        ...parsedResponse,
        metadata: {
          featuresUsed: originalFeatures,
          featuresCount: originalFeatures.length,
          modelType: 'sigmoid_neuron',
          success: true,
        }
      });
      
    } catch (sagemakerError: unknown) {
      const error = sagemakerError as { message?: string; OriginalMessage?: string };
      console.error(`[API DEBUG] SageMaker request failed:`, error?.message || sagemakerError);
      
      if (error?.OriginalMessage) {
        console.error(`[API DEBUG] SageMaker error details:`, error.OriginalMessage);
      }
      
      throw sagemakerError;
    }
    
  } catch (error: unknown) {
    const err = error as { message?: string; stack?: string; OriginalMessage?: string; ErrorCode?: string; name?: string };
    console.error('[API ERROR] ============ CAUGHT EXCEPTION ============');
    console.error('[API ERROR] Error type:', typeof error);
    console.error('[API ERROR] Error name:', err?.name);
    console.error('[API ERROR] Error message:', err?.message);
    console.error('[API ERROR] Full error object:', error);
    console.error('[API ERROR] Error stack:', err?.stack);
    
    if (err?.OriginalMessage) {
      console.error('[API ERROR] SageMaker original message:', err.OriginalMessage);
    }
    
    // Determine if this is a parsing error, AWS error, or other
    let errorCategory = 'unknown';
    if (err?.message?.includes('JSON')) {
      errorCategory = 'json_parsing';
    } else if (err?.message?.includes('AWS') || err?.ErrorCode) {
      errorCategory = 'aws_sagemaker';
    } else if (err?.message?.includes('fetch') || err?.message?.includes('network')) {
      errorCategory = 'network';
    }
    
    console.error('[API ERROR] Error category:', errorCategory);
    
    try {
      const errorResponse = NextResponse.json({ 
        error: 'Failed to fetch prediction from SageMaker.',
        errorCategory: errorCategory,
        debugError: {
          message: err?.message || 'Unknown error',
          originalMessage: err?.OriginalMessage,
          errorCode: err?.ErrorCode,
          name: err?.name,
          stack: err?.stack,
          timestamp: new Date().toISOString(),
          suggestions: [
            'Check browser console and terminal for detailed error logs',
            'Verify environment variables are set correctly',
            'Ensure AWS credentials are valid',
            'Check if SageMaker endpoint is running'
          ]
        }
      }, { status: 500 });
      
      console.log('[API DEBUG] Returning error response:', errorResponse);
      return errorResponse;
    } catch (responseError) {
      console.error('[API ERROR] Failed to create error response:', responseError);
      // Fallback to a simple response if JSON creation fails
      return new Response('Internal Server Error', { status: 500 });
    }
  }
}
