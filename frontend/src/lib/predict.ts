import {
  InvokeEndpointCommand,
  SageMakerRuntimeClient
} from "@aws-sdk/client-sagemaker-runtime";

// ─── SageMaker Runtime Client (singleton) ─────────────────────────────
export const client = new SageMakerRuntimeClient({
  region: process.env.AWS_REGION || "us-west-2",
  credentials: {
    accessKeyId:     process.env.AWS_ACCESS_KEY_ID as string,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY as string,
    sessionToken:    process.env.AWS_SESSION_TOKEN      // may be undefined
  }
});
