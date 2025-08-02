"use client";

import Image from "next/image";
import { useState } from "react";
import { encodeRow } from "@/lib/encoder";

interface FormData {
  // Student Identifier
  rowId: string;
  studentName: string;
  
  // Mental Health
  mentalHealthRating: string; // Excellent, Good, Neutral, Poor, Very poor
  avoidConfrontation: boolean;
  senseOfBelonging: string; // Agree, Neutral, Disagree
  roomRequest: boolean;
  
  // Nutritional Health
  eatingHabits: string; // 0 Times, 1-5 Times, 6-10 Times, More than 10 Times
  eatingAlone: string; // 0 Times, 1-5 Times, 6-10 Times, More than 10 Times
  eatingWithFriends: string; // 0 Times, 1-5 Times, 6-10 Times, More than 10 Times
  preparedMealAlone: string; // 0 Times, 1-5 Times, 6-10 Times, More than 10 Times
  nutritionalConfidence: number; // Keep 1-5 scale
  
  // Job Information
  jobOnCampus: boolean;
  jobOffCampus: boolean;
  internshipOnCampus: boolean;
  internshipOffCampus: boolean;
  undergradResearch: boolean;
  
  // Sense of belonging
  attendedEvent: boolean;
  
  // Other factors
  roommateConflicts: string; // Agree, Neutral, Disagree
  seekingAdvice: string; // Agree, Neutral, Disagree
  
  // Additional Social & Support factors
  seekAssistanceRHC: string; // Agree/Neutral/Disagree
  seekAssistanceRA: string; // Agree/Neutral/Disagree
  useSharedLivingAgreement: string; // Agree/Neutral/Disagree
  seekAdviceFamily: string; // Agree/Neutral/Disagree
  initiateOpenCommunication: string; // Agree/Neutral/Disagree
  roommateRelationship: string; // Satisfied/Neutral/Unsatisfied/Other
  suiteApartmentOccupants: number; // 1-5+
  wellBeingResources: string; // Text response for nutritional resources
}

interface PredictionResult {
  prediction?: number | string | object;
  predictions?: number[][];
  error?: string;
  metadata?: {
    featuresUsed?: number[];
    featuresCount?: number;
    originalFeaturesCount?: number;
    modelType?: string;
    success?: boolean;
  };
  [key: string]: unknown;
}

export default function Home() {
  const [showForm, setShowForm] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null);
  const [formData, setFormData] = useState<FormData>({
    rowId: '',
    studentName: '',
    mentalHealthRating: 'Neutral',
    avoidConfrontation: false,
    senseOfBelonging: 'Neutral',
    roomRequest: false,
    eatingHabits: '1-5 Times',
    eatingAlone: '1-5 Times',
    eatingWithFriends: '1-5 Times',
    preparedMealAlone: '1-5 Times',
    nutritionalConfidence: 3,
    jobOnCampus: false,
    jobOffCampus: false,
    internshipOnCampus: false,
    internshipOffCampus: false,
    undergradResearch: false,
    attendedEvent: false,
    roommateConflicts: 'Neutral',
    seekingAdvice: 'Neutral',
    seekAssistanceRHC: 'Neutral',
    seekAssistanceRA: 'Neutral',
    useSharedLivingAgreement: 'Neutral',
    seekAdviceFamily: 'Neutral',
    initiateOpenCommunication: 'Neutral',
    roommateRelationship: 'Neutral',
    suiteApartmentOccupants: 2,
    wellBeingResources: 'Did not answer',
  });



  // Interpret SageMaker's raw prediction value
  const interpretSageMakerPrediction = (rawValue: number): {
    rawValue: number;
    riskCategory: string;
  } => {
    // The model returns a raw logit value (e.g., 3.698)
    // Higher positive values = higher risk level
    
    let riskCategory: string;
    
    if (rawValue < -1) {
      riskCategory = 'Low Risk';
    } else if (rawValue < 1) {
      riskCategory = 'Moderate Risk';
    } else if (rawValue < 2.5) {
      riskCategory = 'High Risk';
    } else {
      riskCategory = 'Critical Risk';
    }
    
    return {
      rawValue: rawValue,
      riskCategory: riskCategory
    };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log('[FRONTEND DEBUG] ============ FORM SUBMISSION START ============');
    
    if (!formData.rowId.trim()) {
      console.log('[FRONTEND DEBUG] Missing RowID, stopping submission');
      alert('Please enter a Student RowID');
      return;
    }
    
    // Echo form data to console as JSON
    console.log('[FRONTEND DEBUG] Original form data:', JSON.stringify(formData, null, 2));
    
    // Build a row matching the shared schema and encode it
    const schemaRow = {
      'Eat at an on campus eatery (Food trucks, restaurants, food court)': formData.eatingHabits,
      'Prepared a meal on my own': formData.preparedMealAlone,
      'Eat with friends': formData.eatingWithFriends,
      'Eat alone': formData.eatingAlone,
      'how confident are you in preparing a nutritious, healthy meal for yourself?': formData.nutritionalConfidence,
      'Worked a job on campus': formData.jobOnCampus ? 'Yes' : 'No',
      'Worked a job off campus': formData.jobOffCampus ? 'Yes' : 'No',
      'Participated in an internship off campus': formData.internshipOffCampus ? 'Yes' : 'No',
      'Participated in an internship on campus': formData.internshipOnCampus ? 'Yes' : 'No',
      'Participated in undergraduate research': formData.undergradResearch ? 'Yes' : 'No',
      'How would you rate your mental health during the last 30 days?': formData.mentalHealthRating,
      'I avoid conflict or confrontation, so I don\'t address the issue.': formData.avoidConfrontation ? 'Agree' : 'Disagree',
      'Feeling a sense of belonging within the university community (Sense of belonging is the feeling that we have satisfied our emotional need to belong to a community or group because we feel accepted, included, respected, and supported by a group)': formData.senseOfBelonging,
      'My roommate(s) and I do not have any conflicts': formData.roommateConflicts === 'Disagree' ? 'Agree' : 'Disagree',
      'I have submitted a room switch request due to a roommate conflict': formData.roomRequest ? 'Agree' : 'Disagree',
      'I seek assistance from my residence hall coordinator (RHC)': formData.seekAssistanceRHC,
      'I seek assistance from my student leader (RA or CA)': formData.seekAssistanceRA,
      'I use the Shared Living Agreement to guide discussion/conversation': formData.useSharedLivingAgreement,
      'I seek advice from my parents or family': formData.seekAdviceFamily,
      'I initiate open communication and discussion': formData.initiateOpenCommunication,
      'In the last 30 days, how would you describe your relationship with your roommate(s)? - Selected Choice': formData.roommateRelationship,
      'How many total people live in your suite or apartment unit?':
        formData.suiteApartmentOccupants === 5
          ? 'More than 5, including myself'
          : `${formData.suiteApartmentOccupants}, including myself`
    };

    const modelFeatures = encodeRow(schemaRow);
    console.log('[FRONTEND DEBUG] Encoded features:', modelFeatures);
    console.log('[FRONTEND DEBUG] Features array length:', modelFeatures.length);
    
    // Prepare payload for API
    const apiPayload = { features: modelFeatures };
    console.log('[FRONTEND DEBUG] API payload:', JSON.stringify(apiPayload, null, 2));
    
    // Call the prediction API
    try {
      console.log('[FRONTEND DEBUG] ============ CALLING API ============');
      console.log('[FRONTEND DEBUG] Making fetch request to /api/predict...');
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiPayload)
      });
      
      console.log('[FRONTEND DEBUG] Response status:', response.status);
      console.log('[FRONTEND DEBUG] Response ok:', response.ok);
      console.log('[FRONTEND DEBUG] Response headers:', response.headers);
      console.log('[FRONTEND DEBUG] Response type:', response.type);
      
      // Check if response has content
      const responseText = await response.text();
      console.log('[FRONTEND DEBUG] Raw response text:', responseText);
      console.log('[FRONTEND DEBUG] Response text length:', responseText.length);
      
      let prediction;
      try {
        prediction = responseText ? JSON.parse(responseText) : {};
        console.log('[FRONTEND DEBUG] ============ API RESPONSE ============');
        console.log('[FRONTEND DEBUG] Parsed prediction response:', JSON.stringify(prediction, null, 2));
      } catch (parseError) {
        console.error('[FRONTEND ERROR] Failed to parse JSON response:', parseError);
        console.error('[FRONTEND ERROR] Raw text was:', responseText);
        prediction = { error: 'Invalid JSON response', rawText: responseText };
      }
      
      if (!response.ok) {
        console.error('[FRONTEND ERROR] API returned error status:', response.status);
        console.error('[FRONTEND ERROR] Error details:', prediction);
        console.error('[FRONTEND ERROR] Response was empty?', Object.keys(prediction).length === 0);
        alert(`API Error (${response.status}): ${prediction.error || 'Unknown error - check console for details'}`);
      } else {
        console.log('[FRONTEND DEBUG] API call successful!');
        
        // Store prediction result for dashboard
        setPredictionResult(prediction);
      }
      
    } catch (error) {
      console.error('[FRONTEND ERROR] ============ FETCH ERROR ============');
      console.error('[FRONTEND ERROR] Network/fetch error:', error);
      alert('Network error: Failed to reach prediction API. Check console for details.');
    }
    
    console.log('[FRONTEND DEBUG] ============ SHOWING DASHBOARD ============');
    setShowDashboard(true);
  };

  const handleInputChange = (field: keyof FormData, value: number | boolean | string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  if (showDashboard) {
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header with SDSU Logo */}
        <header className="bg-white shadow-sm p-6 flex justify-between items-center">
          <Image
            src="/San-Diego-State-University-Logo-removebg-preview.png"
            alt="San Diego State University Logo"
            width={120}
            height={80}
            className="object-contain"
          />
          <button
            onClick={() => setShowDashboard(false)}
            className="px-4 py-2 bg-primary-red text-white font-regular rounded-lg hover:bg-red-hover transition-colors"
          >
            Edit Analysis
          </button>
        </header>

        <main className="max-w-6xl mx-auto p-8">
          <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
            <div className="flex justify-between items-start mb-8">
              <div>
                <h1 className="text-4xl font-regular text-gray-900 mb-2">Student Risk Analysis</h1>
                <p className="text-xl font-light text-gray-600">AI Prediction Model Results</p>
                {predictionResult && predictionResult.predictions && predictionResult.predictions[0] && (
                  <div className={`mt-4 p-4 rounded-lg ${
                    (1 / (1 + Math.exp(-predictionResult.predictions[0][0])) * 100) > 70 
                      ? 'bg-red-100 border border-primary-red' 
                      : (1 / (1 + Math.exp(-predictionResult.predictions[0][0])) * 100) > 40 
                        ? 'bg-yellow-100 border border-yellow-500' 
                        : 'bg-green-100 border border-green-500'
                  }`}>
                    <p className="text-2xl font-regular text-black">
                      Attrition Risk: {(1 / (1 + Math.exp(-predictionResult.predictions[0][0])) * 100).toFixed(1)}%
                    </p>
                  </div>
                )}
              </div>
              <div className="text-right">
                <p className="text-sm font-light text-gray-500">Student RowID</p>
                <p className="text-2xl font-regular text-primary-red">{formData.rowId}</p>
                {formData.studentName && (
                  <>
                    <p className="text-sm font-light text-gray-500 mt-2">Student Name</p>
                    <p className="text-lg font-regular text-gray-900">{formData.studentName}</p>
                  </>
                )}
              </div>
            </div>
            


            {/* Assessment Results Grid */}
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Mental Health */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-xl font-regular text-primary-red mb-4">Mental Health Data</h3>
                <div className="space-y-2 font-light text-black">
                  <p>Mental Health Rating: {formData.mentalHealthRating}</p>
                  <p>Tends to Avoid Confrontation: {formData.avoidConfrontation ? 'Yes' : 'No'}</p>
                  <p>Sense of Belonging: {formData.senseOfBelonging}</p>
                  <p>Room Change Requests: {formData.roomRequest ? 'Yes' : 'No'}</p>
                </div>
              </div>

              {/* Nutritional Health */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-xl font-regular text-primary-red mb-4">Nutritional Health Data</h3>
                <div className="space-y-2 font-light text-black">
                  <p>Campus Eatery Frequency: {formData.eatingHabits}</p>
                  <p>Eating Alone Frequency: {formData.eatingAlone}</p>
                  <p>Eating with Friends Frequency: {formData.eatingWithFriends}</p>
                  <p>Preparing Meals Alone Frequency: {formData.preparedMealAlone}</p>
                  <p>Nutritional Meal Confidence: {formData.nutritionalConfidence}/5</p>
                </div>
              </div>

              {/* Job Information */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-xl font-regular text-primary-red mb-4">Employment & Research Data</h3>
                <div className="space-y-2 font-light text-black">
                  <p>Has Job On Campus: {formData.jobOnCampus ? 'Yes' : 'No'}</p>
                  <p>Has Job Off Campus: {formData.jobOffCampus ? 'Yes' : 'No'}</p>
                  <p>Has Internship On Campus: {formData.internshipOnCampus ? 'Yes' : 'No'}</p>
                  <p>Has Internship Off Campus: {formData.internshipOffCampus ? 'Yes' : 'No'}</p>
                  <p>Involved in Undergrad Research: {formData.undergradResearch ? 'Yes' : 'No'}</p>
                </div>
              </div>

              {/* Social Engagement */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-xl font-regular text-primary-red mb-4">Social Engagement Data</h3>
                <div className="space-y-2 font-light text-black">
                  <p>Has Attended Campus Events: {formData.attendedEvent ? 'Yes' : 'No'}</p>
                  <p>Has Roommate Conflicts: {formData.roommateConflicts}</p>
                  <p>Seeks Advice from Parents/Family: {formData.seekingAdvice}</p>
                  <p>Seeks Assistance from RHC: {formData.seekAssistanceRHC}</p>
                  <p>Seeks Assistance from RA/CA: {formData.seekAssistanceRA}</p>
                  <p>Uses Shared Living Agreement: {formData.useSharedLivingAgreement}</p>
                  <p>Seeks Advice from Family: {formData.seekAdviceFamily}</p>
                  <p>Initiates Open Communication: {formData.initiateOpenCommunication}</p>
                  <p>Roommate Relationship (30 days): {formData.roommateRelationship}</p>
                  <p>Suite/Apartment Occupants: {formData.suiteApartmentOccupants === 5 ? '5+' : formData.suiteApartmentOccupants}</p>
                  <p>Well-being Resources: {formData.wellBeingResources}</p>
                </div>
              </div>
            </div>

            {/* SageMaker Prediction Results */}
            {predictionResult && (
              <div className="bg-blue-50 rounded-xl shadow-lg p-8 mt-8">
                <h2 className="text-3xl font-regular text-blue-700 mb-4">AI Model Prediction</h2>
                
                {/* Interpreted Results */}
                {predictionResult.predictions && predictionResult.predictions[0] && predictionResult.predictions[0][0] !== undefined && (
                  <div className="grid md:grid-cols-2 gap-6 mb-6">
                    <div className="bg-white rounded-lg p-6">
                      <h3 className="text-xl font-regular text-blue-700 mb-4">Model Output</h3>
                      <div className="space-y-3">
                        {(() => {
                          const rawValue = predictionResult.predictions![0][0];
                          const interpretation = interpretSageMakerPrediction(rawValue);
                          return (
                            <>
                              <div className="flex justify-between">
                                <span className="font-light text-gray-600">Raw Model Output:</span>
                                <span className="font-regular text-black">{rawValue.toFixed(4)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="font-light text-gray-600">Sigmoid Probability:</span>
                                <span className="font-regular text-black">{(1 / (1 + Math.exp(-rawValue)) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="font-light text-gray-600">Risk Assessment:</span>
                                <span className={`font-regular ${
                                  interpretation.riskCategory === 'Critical Risk' ? 'text-primary-red' : 
                                  interpretation.riskCategory === 'High Risk' ? 'text-primary-red' : 
                                  interpretation.riskCategory === 'Moderate Risk' ? 'text-yellow-600' : 
                                  'text-green-600'
                                }`}>
                                  {interpretation.riskCategory}
                                </span>
                              </div>
                            </>
                          );
                        })()}
                      </div>
                    </div>
                    
                    <div className="bg-white rounded-lg p-6">
                      <h3 className="text-xl font-regular text-blue-700 mb-4">Model Metadata</h3>
                      <div className="space-y-3">
                        {predictionResult.metadata && (
                          <>
                            <div className="flex justify-between">
                              <span className="font-light text-gray-600">Features Used:</span>
                              <span className="font-regular text-black">{predictionResult.metadata.featuresCount || 'N/A'}/22</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="font-light text-gray-600">Model Type:</span>
                              <span className="font-regular text-black capitalize">{predictionResult.metadata.modelType?.replace('_', ' ') || 'Unknown'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="font-light text-gray-600">Status:</span>
                              <span className="font-regular text-green-600">Success</span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                )}
                
                {/* Raw JSON for debugging */}
                <details className="bg-white rounded-lg p-6">
                  <summary className="font-regular text-blue-700 cursor-pointer mb-4">
                    Raw API Response (for debugging)
                  </summary>
                  <pre className="font-light text-black overflow-auto text-sm bg-gray-50 p-4 rounded">
                    {JSON.stringify(predictionResult, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        </main>
      </div>
    );
  
  }

  if (showForm) {
    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header with SDSU Logo */}
        <header className="bg-white shadow-sm p-6">
          <Image
            src="/San-Diego-State-University-Logo-removebg-preview.png"
            alt="San Diego State University Logo"
            width={120}
            height={80}
            className="object-contain"
          />
        </header>

        <main className="max-w-4xl mx-auto p-8">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h1 className="text-4xl font-regular text-gray-900 mb-2">Student Data Analysis</h1>
            <p className="text-xl font-light text-gray-600 mb-8">Enter student information for AI risk prediction</p>
            
            <form onSubmit={handleSubmit} className="space-y-8">
              {/* Student Identification Section */}
              <div className="border-l-4 border-primary-red pl-6">
                <h2 className="text-2xl font-regular text-primary-red mb-6">Student Identification</h2>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student RowID *</label>
                    <input
                      type="text"
                      value={formData.rowId}
                      onChange={(e) => handleInputChange('rowId', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                      placeholder="Enter student RowID"
                      required
                    />
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student Name (Optional)</label>
                    <input
                      type="text"
                      value={formData.studentName}
                      onChange={(e) => handleInputChange('studentName', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                      placeholder="Enter student name"
                    />
                  </div>
                </div>
              </div>
              {/* Mental Health Section */}
              <div className="border-l-4 border-primary-red pl-6">
                <h2 className="text-2xl font-regular text-primary-red mb-6">Mental Health Data</h2>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student&apos;s mental health rating</label>
                    <select
                      value={formData.mentalHealthRating}
                      onChange={(e) => handleInputChange('mentalHealthRating', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Excellent">Excellent</option>
                      <option value="Good">Good</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Poor">Poor</option>
                      <option value="Very poor">Very poor</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student&apos;s sense of belonging</label>
                    <select
                      value={formData.senseOfBelonging}
                      onChange={(e) => handleInputChange('senseOfBelonging', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.avoidConfrontation}
                      onChange={(e) => handleInputChange('avoidConfrontation', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student tends to avoid confrontation</label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.roomRequest}
                      onChange={(e) => handleInputChange('roomRequest', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student has made room change requests</label>
                  </div>
                </div>
              </div>

              {/* Nutritional Health Section */}
              <div className="border-l-4 border-primary-red pl-6">
                <h2 className="text-2xl font-regular text-primary-red mb-6">Nutritional Health Data</h2>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">How often student eats out on campus eatery</label>
                    <select
                      value={formData.eatingHabits}
                      onChange={(e) => handleInputChange('eatingHabits', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="0 Times">0 Times</option>
                      <option value="1-5 Times">1-5 Times</option>
                      <option value="6-10 Times">6-10 Times</option>
                      <option value="More than 10 Times">More than 10 Times</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">How often student eats alone</label>
                    <select
                      value={formData.eatingAlone}
                      onChange={(e) => handleInputChange('eatingAlone', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="0 Times">0 Times</option>
                      <option value="1-5 Times">1-5 Times</option>
                      <option value="6-10 Times">6-10 Times</option>
                      <option value="More than 10 Times">More than 10 Times</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">How often student eats with friends</label>
                    <select
                      value={formData.eatingWithFriends}
                      onChange={(e) => handleInputChange('eatingWithFriends', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="0 Times">0 Times</option>
                      <option value="1-5 Times">1-5 Times</option>
                      <option value="6-10 Times">6-10 Times</option>
                      <option value="More than 10 Times">More than 10 Times</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">How often student prepares meals alone</label>
                    <select
                      value={formData.preparedMealAlone}
                      onChange={(e) => handleInputChange('preparedMealAlone', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="0 Times">0 Times</option>
                      <option value="1-5 Times">1-5 Times</option>
                      <option value="6-10 Times">6-10 Times</option>
                      <option value="More than 10 Times">More than 10 Times</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student&apos;s confidence in preparing nutritious meals (1-5)</label>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={formData.nutritionalConfidence}
                      onChange={(e) => handleInputChange('nutritionalConfidence', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.nutritionalConfidence}</span>
                  </div>
                </div>
              </div>

              {/* Job Information Section */}
              <div className="border-l-4 border-primary-red pl-6">
                <h2 className="text-2xl font-regular text-primary-red mb-6">Employment & Research Data</h2>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.jobOnCampus}
                      onChange={(e) => handleInputChange('jobOnCampus', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student has a job on campus</label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.jobOffCampus}
                      onChange={(e) => handleInputChange('jobOffCampus', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student has a job off campus</label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.internshipOnCampus}
                      onChange={(e) => handleInputChange('internshipOnCampus', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student has an internship on campus</label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.internshipOffCampus}
                      onChange={(e) => handleInputChange('internshipOffCampus', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student has an internship off campus</label>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.undergradResearch}
                      onChange={(e) => handleInputChange('undergradResearch', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student is involved in undergraduate research</label>
                  </div>
                </div>
              </div>

              {/* Additional Factors Section */}
              <div className="border-l-4 border-primary-red pl-6">
                <h2 className="text-2xl font-regular text-primary-red mb-6">Social & Support Data</h2>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.attendedEvent}
                      onChange={(e) => handleInputChange('attendedEvent', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student has attended campus events</label>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student has roommate conflicts</label>
                    <select
                      value={formData.roommateConflicts}
                      onChange={(e) => handleInputChange('roommateConflicts', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student seeks advice from parents or family</label>
                    <select
                      value={formData.seekingAdvice}
                      onChange={(e) => handleInputChange('seekingAdvice', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student seeks assistance from residence hall coordinator (RHC)</label>
                    <select
                      value={formData.seekAssistanceRHC}
                      onChange={(e) => handleInputChange('seekAssistanceRHC', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student seeks assistance from student leader (RA or CA)</label>
                    <select
                      value={formData.seekAssistanceRA}
                      onChange={(e) => handleInputChange('seekAssistanceRA', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student uses Shared Living Agreement to guide discussions</label>
                    <select
                      value={formData.useSharedLivingAgreement}
                      onChange={(e) => handleInputChange('useSharedLivingAgreement', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student seeks advice from parents or family</label>
                    <select
                      value={formData.seekAdviceFamily}
                      onChange={(e) => handleInputChange('seekAdviceFamily', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student initiates open communication and discussion</label>
                    <select
                      value={formData.initiateOpenCommunication}
                      onChange={(e) => handleInputChange('initiateOpenCommunication', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Agree">Agree</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Disagree">Disagree</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student&apos;s relationship with roommate(s) in last 30 days</label>
                    <select
                      value={formData.roommateRelationship}
                      onChange={(e) => handleInputChange('roommateRelationship', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Satisfied">Satisfied</option>
                      <option value="Neutral">Neutral</option>
                      <option value="Unsatisfied">Unsatisfied</option>
                      <option value="Other">Other</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Total people living in suite/apartment unit</label>
                    <select
                      value={formData.suiteApartmentOccupants}
                      onChange={(e) => handleInputChange('suiteApartmentOccupants', parseInt(e.target.value))}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value={1}>1</option>
                      <option value={2}>2</option>
                      <option value={3}>3</option>
                      <option value={4}>4</option>
                      <option value={5}>5+</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Well-being & Health Promotion for nutritional resources</label>
                    <select
                      value={formData.wellBeingResources}
                      onChange={(e) => handleInputChange('wellBeingResources', e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-red focus:border-transparent font-light text-black"
                    >
                      <option value="Did not answer">Did not answer</option>
                      <option value="No, I was not aware of these resources">No, I was not aware of these resources</option>
                      <option value="No, I prefer to handle things on my own">No, I prefer to handle things on my own</option>
                      <option value="Yes, I have used these resources">Yes, I have used these resources</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="flex gap-4 pt-6">
                                  <button
                    type="submit"
                    className="px-8 py-3 bg-primary-red text-white font-regular rounded-lg hover:bg-red-hover transition-colors"
                  >
                    Generate AI Risk Analysis
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowForm(false)}
                    className="px-8 py-3 bg-gray-300 text-gray-700 font-regular rounded-lg hover:bg-gray-400 transition-colors"
                  >
                    Back to Portal
                  </button>
              </div>
            </form>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen">
      {/* Header with SDSU Logo */}
      <header className="absolute top-0 left-0 z-10 p-6">
        <Image
          src="/San-Diego-State-University-Logo-removebg-preview.png"
          alt="San Diego State University Logo"
          width={120}
          height={80}
          priority
          className="object-contain"
        />
      </header>

      {/* Main Content with Banner */}
      <main className="relative min-h-screen flex items-center justify-center">
        {/* Background Banner Image */}
        <div className="absolute inset-0 z-0">
          <Image
            src="/newsitebannerv6.png"
            alt="Student Success Portal Banner"
            fill
            className="object-cover"
            priority
          />
          {/* Overlay for better text readability */}
          <div className="absolute inset-0 bg-black/30"></div>
        </div>

        {/* Content */}
        <div className="relative z-10 text-center text-white px-6 max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-7xl font-normal mb-6 leading-tight">
            Welcome to Student Success Portal
          </h1>
          <p className="text-xl md:text-2xl font-light max-w-2xl mx-auto leading-relaxed mb-8">
            Monitor student progress, identify risks, and drive positive outcomes
          </p>
          <p className="text-lg font-light max-w-xl mx-auto mb-8 opacity-90">
            Administrative AI Prediction Interface for Student Attrition Risk Analysis
          </p>
          
          <button
            onClick={() => setShowForm(true)}
            className="px-8 py-4 bg-primary-red text-white font-regular text-lg rounded-lg hover:bg-red-hover transition-colors"
          >
            Analyze Student Data
          </button>
        </div>
      </main>
    </div>
  );
}

