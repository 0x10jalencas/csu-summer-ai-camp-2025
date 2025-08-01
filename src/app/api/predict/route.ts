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

    const region = process.env.AWS_REGION;
    const endpointName = process.env.SAGEMAKER_ENDPOINT_NAME;

    if (!region || !endpointName) {
      return NextResponse.json(
        {
          error: 'AWS_REGION and SAGEMAKER_ENDPOINT_NAME must be configured in environment variables.',
        },
        { status: 500 }
      );
    }

    // Initialize SageMaker Runtime client
    const client = new SageMakerRuntimeClient({ region });

    // Prepare the command to invoke the endpoint
    const command = new InvokeEndpointCommand({
      EndpointName: endpointName,
      Body: Buffer.from(JSON.stringify(payload)),
      ContentType: 'application/json',
      Accept: 'application/json',
    });

    // Send the request to SageMaker
    const response = await client.send(command);

    // Convert the binary body back to string
    const responseBody = new TextDecoder('utf-8').decode(response.Body as Uint8Array);

    // Return the parsed JSON response back to the frontend
    return NextResponse.json(JSON.parse(responseBody));
  } catch (error) {
    console.error('SageMaker invocation error:', error);
    return NextResponse.json({ error: 'Failed to fetch prediction from SageMaker.' }, { status: 500 });
  }
}
