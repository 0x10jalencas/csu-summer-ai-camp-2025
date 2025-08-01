import { NextRequest, NextResponse } from 'next/server';
import { InvokeEndpointCommand, SageMakerRuntimeClient } from '@aws-sdk/client-sagemaker-runtime';

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
  try {
    // Parse incoming JSON payload from the request body
    const payload = await req.json();
    
    console.log('🔍 [API DEBUG] ============ PREDICTION REQUEST DEBUG ============');
    console.log('🔍 [API DEBUG] Received payload:', JSON.stringify(payload, null, 2));
    console.log('🔍 [API DEBUG] Payload type:', typeof payload);
    console.log('🔍 [API DEBUG] Payload keys:', Object.keys(payload));
    
    if (payload.features) {
      console.log('🔍 [API DEBUG] Features array found!');
      console.log('🔍 [API DEBUG] Features array length:', payload.features.length);
      console.log('🔍 [API DEBUG] Features array:', payload.features);
      console.log('🔍 [API DEBUG] Features types:', payload.features.map((f: unknown) => typeof f));
      console.log('🔍 [API DEBUG] All features are numbers?', payload.features.every((f: unknown) => typeof f === 'number'));
    } else {
      console.log('🔍 [API DEBUG] ❌ NO FEATURES KEY FOUND IN PAYLOAD!');
    }

    const region = process.env.AWS_REGION;
    const endpointName = process.env.SAGEMAKER_ENDPOINT_NAME ?? 'pytorch-inference-2025-08-01-02-22-44-558';

    console.log('🔍 [API DEBUG] AWS Region:', region);
    console.log('🔍 [API DEBUG] Endpoint Name:', endpointName);
    console.log('🔍 [API DEBUG] Mock mode?', process.env.MOCK_PREDICTION);

    if (!region) {
      return NextResponse.json(
        {
          error: 'AWS_REGION must be configured in environment variables.',
        },
        { status: 500 }
      );
    }

    // If MOCK_PREDICTION env var is set, skip the real invocation and return a mock
    if (process.env.MOCK_PREDICTION === 'true') {
      console.log('🔍 [API DEBUG] Returning mock prediction because MOCK_PREDICTION=true');
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

    // Initialize SageMaker Runtime client
    const client = new SageMakerRuntimeClient({ region });

    // CONFIRMED: Model expects exactly 10 features (based on successful testing)
    console.log('🔍 [API DEBUG] ============ SENDING TO SAGEMAKER ============');

    const originalFeatures = payload.features || [];
    console.log('🔍 [API DEBUG] Original features length:', originalFeatures.length);
    
    // Use only first 10 features (the ones the model was trained with)
    const modelFeatures = originalFeatures.slice(0, 10);
    console.log('🔍 [API DEBUG] Using first 10 features:', modelFeatures);
    
    // Send as direct array (the format that worked in testing)
    const requestData = modelFeatures;

    try {
      const command = new InvokeEndpointCommand({
        EndpointName: endpointName,
        Body: Buffer.from(JSON.stringify(requestData)),
        ContentType: 'application/json',
        Accept: 'application/json',
      });

      console.log('🔍 [API DEBUG] Sending request to SageMaker...');
      const response = await client.send(command);

      // Convert the binary body back to string
      const responseBody = new TextDecoder('utf-8').decode(response.Body as Uint8Array);
      console.log('🔍 [API DEBUG] ✅ SUCCESS! Raw response body:', responseBody);

      const parsedResponse = JSON.parse(responseBody);
      console.log('🔍 [API DEBUG] ✅ SUCCESS! Parsed response:', parsedResponse);
      
      // Success! Return the result with metadata
      return NextResponse.json({
        ...parsedResponse,
        metadata: {
          featuresUsed: modelFeatures,
          featuresCount: modelFeatures.length,
          originalFeaturesCount: originalFeatures.length,
          modelType: 'sigmoid_neuron',
          success: true
        }
      });
      
    } catch (sagemakerError: unknown) {
      const error = sagemakerError as { message?: string; OriginalMessage?: string };
      console.error(`🔍 [API DEBUG] ❌ SageMaker request failed:`, error?.message || sagemakerError);
      
      if (error?.OriginalMessage) {
        console.error(`🔍 [API DEBUG] SageMaker error details:`, error.OriginalMessage);
      }
      
      throw sagemakerError;
    }
    
  } catch (error: unknown) {
    const err = error as { message?: string; stack?: string; OriginalMessage?: string; ErrorCode?: string };
    console.error('🚨 [API ERROR] ============ ALL ATTEMPTS FAILED ============');
    console.error('🚨 [API ERROR] Final error:', error);
    console.error('🚨 [API ERROR] Error message:', err?.message);
    console.error('🚨 [API ERROR] Error stack:', err?.stack);
    
    if (err?.OriginalMessage) {
      console.error('🚨 [API ERROR] SageMaker original message:', err.OriginalMessage);
    }
    
    return NextResponse.json({ 
      error: 'Failed to fetch prediction from SageMaker.',
      debugError: {
        message: err?.message || 'Unknown error',
        originalMessage: err?.OriginalMessage,
        errorCode: err?.ErrorCode,
        stack: err?.stack,
        suggestions: [
          'Model expects 10 features based on matrix multiplication error',
          'Check if model was trained with different feature set',
          'Verify feature encoding matches training data'
        ]
      }
    }, { status: 500 });
  }
}
