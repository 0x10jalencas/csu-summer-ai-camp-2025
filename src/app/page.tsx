"use client";

import Image from "next/image";
import { useState } from "react";

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

  const calculateAttritionProbability = (data: FormData): number => {
    let riskScore = 0;
    
    // Mental health factors (higher weight)
    const getMentalHealthRisk = (rating: string) => {
      const riskMap = {
        'Excellent': 0,
        'Good': 2,
        'Neutral': 5,
        'Poor': 8,
        'Very poor': 12
      };
      return riskMap[rating as keyof typeof riskMap] || 5;
    };
    
    riskScore += getMentalHealthRisk(data.mentalHealthRating);
    riskScore += data.avoidConfrontation ? 3 : 0;
    
    const getSenseOfBelongingRisk = (rating: string) => {
      const riskMap = {
        'Agree': 0,
        'Neutral': 3,
        'Disagree': 6
      };
      return riskMap[rating as keyof typeof riskMap] || 3;
    };
    
    riskScore += getSenseOfBelongingRisk(data.senseOfBelonging);
    riskScore += data.roomRequest ? 2 : 0;
    
    // Nutritional health
    // Convert frequency strings to risk scores
    const getFrequencyRisk = (frequency: string, isPositive: boolean = false) => {
      const riskMap = {
        '0 Times': isPositive ? 3 : 0,
        '1-5 Times': isPositive ? 2 : 1,
        '6-10 Times': isPositive ? 1 : 2,
        'More than 10 Times': isPositive ? 0 : 3
      };
      return riskMap[frequency as keyof typeof riskMap] || 1;
    };
    
    riskScore += getFrequencyRisk(data.eatingHabits, false); // Poor eating habits = risk
    riskScore += getFrequencyRisk(data.eatingAlone, false); // Eating alone frequently = risk
    riskScore += getFrequencyRisk(data.eatingWithFriends, true); // Eating with friends frequently = protective
    riskScore += getFrequencyRisk(data.preparedMealAlone, false); // Preparing meals alone frequently = risk
    riskScore += (5 - data.nutritionalConfidence) * 1.5;
    
    // Job/engagement factors (protective)
    riskScore += data.jobOnCampus ? -3 : 1;
    riskScore += data.jobOffCampus ? -1 : 0;
    riskScore += data.internshipOnCampus ? -4 : 0;
    riskScore += data.internshipOffCampus ? -2 : 0;
    riskScore += data.undergradResearch ? -5 : 0;
    
    // Social engagement
    riskScore += data.attendedEvent ? -2 : 2;
    
    // Conflict and support
    const getRoommateConflictRisk = (rating: string) => {
      const riskMap = {
        'Disagree': 0, // No conflicts = no risk
        'Neutral': 3,  // Some conflicts = moderate risk
        'Agree': 6     // Has conflicts = higher risk
      };
      return riskMap[rating as keyof typeof riskMap] || 3;
    };
    
    riskScore += getRoommateConflictRisk(data.roommateConflicts);
    
    const getSeekingAdviceRisk = (rating: string) => {
      const riskMap = {
        'Agree': 0,     // Seeks advice from family = protective
        'Neutral': 2,   // Sometimes seeks advice = moderate
        'Disagree': 4   // Doesn't seek advice = higher risk
      };
      return riskMap[rating as keyof typeof riskMap] || 2;
    };
    
    riskScore += getSeekingAdviceRisk(data.seekingAdvice);
    
    // Additional social support factors
    riskScore += data.seekAssistanceRHC === 'Disagree' ? 2 : data.seekAssistanceRHC === 'Agree' ? -1 : 0;
    riskScore += data.seekAssistanceRA === 'Disagree' ? 2 : data.seekAssistanceRA === 'Agree' ? -1 : 0;
    riskScore += data.useSharedLivingAgreement === 'Disagree' ? 1.5 : data.useSharedLivingAgreement === 'Agree' ? -1 : 0;
    riskScore += data.seekAdviceFamily === 'Disagree' ? 3 : data.seekAdviceFamily === 'Agree' ? -2 : 0;
    riskScore += data.initiateOpenCommunication === 'Disagree' ? 2.5 : data.initiateOpenCommunication === 'Agree' ? -1.5 : 0;
    riskScore += data.roommateRelationship === 'Unsatisfied' ? 4 : data.roommateRelationship === 'Satisfied' ? -2 : 0;
    riskScore += data.suiteApartmentOccupants === 1 ? 2 : data.suiteApartmentOccupants >= 5 ? 1 : 0;
    
    // Convert to probability (0-100%)
    const probability = Math.max(0, Math.min(100, (riskScore / 50) * 100));
    return Math.round(probability * 10) / 10;
  };

  // Transform form data to model's expected 10-feature array
  const transformToModelFeatures = (data: FormData): number[] => {
    console.log('🔍 [FRONTEND DEBUG] ============ FEATURE TRANSFORMATION DEBUG ============');
    console.log('🔍 [FRONTEND DEBUG] Input form data:', JSON.stringify(data, null, 2));
    // Feature mappings based on LabelEncoder alphabetical sorting
    // NOTE: Model uses only first 10 features (eating patterns + employment/activities + mental health)
    
    // 1-4: Frequency features (eating patterns)
    const getFrequencyCode = (freq: string): number => {
      const map: { [key: string]: number } = {
        '0 Times': 0,
        '1-5 Times': 1,
        '6-10 Times': 2,
        'Did not answer': 3,
        'More than 10 Times': 4
      };
      return map[freq] ?? 3; // Default to "Did not answer"
    };

    // 6-10: Employment/Activities
    const getBooleanCode = (value: boolean): number => {
      // "Did not answer"→0, "No"→1, "Not Applicable"→2, "Yes"→3
      return value ? 3 : 1; // Yes→3, No→1
    };

    // 11: Mental Health
    const getMentalHealthCode = (rating: string): number => {
      const map: { [key: string]: number } = {
        'Did not answer': 0,
        'Excellent': 1,
        'Good': 2,
        'Neutral': 3,
        'Poor': 4,
        'Very poor': 5
      };
      return map[rating] ?? 0;
    };

    // 12-20: Agree/Disagree features
    const getAgreeDisagreeCode = (value: string): number => {
      const map: { [key: string]: number } = {
        'Agree': 0,
        'Did not answer': 1,
        'Disagree': 2,
        'Neutral': 3
      };
      return map[value] ?? 1;
    };

    // 21: Roommate relationship
    const getRoommateRelationshipCode = (value: string): number => {
      const map: { [key: string]: number } = {
        'Did not answer': 0,
        'Neutral': 1, // "Neutral, I don't interact with my roommates, but we can coexist in the same space."
        'Other': 2,   // "Other, please specify"
        'Satisfied': 3, // "Satisfied, most of us get along well and we usually work together to resolve conflicts."
        'Unsatisfied': 4 // "Unsatisfied, most of us don't get along well, and we usually are not able to resolve conflicts."
      };
      return map[value] ?? 0;
    };

    // 22: Suite occupants
    const getSuiteOccupantsCode = (count: number): number => {
      const map: { [key: number]: number } = {
        1: 0, // "1, only me"
        2: 1, // "2, including myself"
        3: 2, // "3, including myself"
        4: 3, // "4, including myself"
        5: 5  // "More than 5, including myself" (note: 4 is "Did not answer")
      };
      return map[count] ?? 4; // Default to "Did not answer"
    };

    // 23: Well-being resources (for now, default to "Did not answer" = 0)
    const getWellBeingCode = (response: string): number => {
      // This would need the full 127-item mapping, for now default to "Did not answer"
      return response === 'Did not answer' ? 0 : 0; // Placeholder
    };

    // Calculate all features with debugging
    const features = [
      getFrequencyCode(data.eatingHabits),                    // 1
      getFrequencyCode(data.preparedMealAlone),               // 2
      getFrequencyCode(data.eatingWithFriends),               // 3
      getFrequencyCode(data.eatingAlone),                     // 4
      data.nutritionalConfidence,                             // 5 (numerical, no encoding)
      getBooleanCode(data.jobOnCampus),                       // 6
      getBooleanCode(data.jobOffCampus),                      // 7
      getBooleanCode(data.internshipOffCampus),               // 8
      getBooleanCode(data.internshipOnCampus),                // 9
      getBooleanCode(data.undergradResearch),                 // 10
      getMentalHealthCode(data.mentalHealthRating),           // 11
      getAgreeDisagreeCode(data.avoidConfrontation ? 'Agree' : 'Disagree'), // 12
      getAgreeDisagreeCode(data.senseOfBelonging),            // 13
      getAgreeDisagreeCode(data.roommateConflicts === 'Disagree' ? 'Agree' : 'Disagree'), // 14 (inverted)
      getAgreeDisagreeCode(data.roomRequest ? 'Agree' : 'Disagree'), // 15
      getAgreeDisagreeCode(data.seekAssistanceRHC),           // 16
      getAgreeDisagreeCode(data.seekAssistanceRA),            // 17
      getAgreeDisagreeCode(data.useSharedLivingAgreement),    // 18
      getAgreeDisagreeCode(data.seekAdviceFamily),            // 19
      getAgreeDisagreeCode(data.initiateOpenCommunication),   // 20
      getRoommateRelationshipCode(data.roommateRelationship), // 21
      getSuiteOccupantsCode(data.suiteApartmentOccupants),    // 22
      getWellBeingCode(data.wellBeingResources)               // 23
    ];

    console.log('🔍 [FRONTEND DEBUG] Generated features array (length=' + features.length + '):', features);
    console.log('🔍 [FRONTEND DEBUG] Feature mapping breakdown:');
    console.log('  1. Eating habits (' + data.eatingHabits + ') → ' + getFrequencyCode(data.eatingHabits));
    console.log('  2. Prepared meal alone (' + data.preparedMealAlone + ') → ' + getFrequencyCode(data.preparedMealAlone));
    console.log('  3. Eating with friends (' + data.eatingWithFriends + ') → ' + getFrequencyCode(data.eatingWithFriends));
    console.log('  4. Eating alone (' + data.eatingAlone + ') → ' + getFrequencyCode(data.eatingAlone));
    console.log('  5. Nutritional confidence (' + data.nutritionalConfidence + ') → ' + data.nutritionalConfidence);
    console.log('  6. Job on campus (' + data.jobOnCampus + ') → ' + getBooleanCode(data.jobOnCampus));
    console.log('  7. Job off campus (' + data.jobOffCampus + ') → ' + getBooleanCode(data.jobOffCampus));
    console.log('  8. Internship off campus (' + data.internshipOffCampus + ') → ' + getBooleanCode(data.internshipOffCampus));
    console.log('  9. Internship on campus (' + data.internshipOnCampus + ') → ' + getBooleanCode(data.internshipOnCampus));
    console.log(' 10. Undergrad research (' + data.undergradResearch + ') → ' + getBooleanCode(data.undergradResearch));
    console.log(' 11. Mental health (' + data.mentalHealthRating + ') → ' + getMentalHealthCode(data.mentalHealthRating));
    console.log(' 12. Avoid confrontation (' + data.avoidConfrontation + ') → ' + getAgreeDisagreeCode(data.avoidConfrontation ? 'Agree' : 'Disagree'));
    console.log(' 13. Sense of belonging (' + data.senseOfBelonging + ') → ' + getAgreeDisagreeCode(data.senseOfBelonging));
    console.log(' 14. Roommate conflicts [INVERTED] (' + data.roommateConflicts + ') → ' + getAgreeDisagreeCode(data.roommateConflicts === 'Disagree' ? 'Agree' : 'Disagree'));
    console.log(' 15. Room request (' + data.roomRequest + ') → ' + getAgreeDisagreeCode(data.roomRequest ? 'Agree' : 'Disagree'));
    console.log(' 16. Seek assistance RHC (' + data.seekAssistanceRHC + ') → ' + getAgreeDisagreeCode(data.seekAssistanceRHC));
    console.log(' 17. Seek assistance RA (' + data.seekAssistanceRA + ') → ' + getAgreeDisagreeCode(data.seekAssistanceRA));
    console.log(' 18. Shared living agreement (' + data.useSharedLivingAgreement + ') → ' + getAgreeDisagreeCode(data.useSharedLivingAgreement));
    console.log(' 19. Seek advice family (' + data.seekAdviceFamily + ') → ' + getAgreeDisagreeCode(data.seekAdviceFamily));
    console.log(' 20. Initiate open communication (' + data.initiateOpenCommunication + ') → ' + getAgreeDisagreeCode(data.initiateOpenCommunication));
    console.log(' 21. Roommate relationship (' + data.roommateRelationship + ') → ' + getRoommateRelationshipCode(data.roommateRelationship));
    console.log(' 22. Suite occupants (' + data.suiteApartmentOccupants + ') → ' + getSuiteOccupantsCode(data.suiteApartmentOccupants));
    console.log(' 23. Well-being resources (' + data.wellBeingResources + ') → ' + getWellBeingCode(data.wellBeingResources));

    console.log('🔍 [FRONTEND DEBUG] All feature types:', features.map(f => typeof f));
    console.log('🔍 [FRONTEND DEBUG] All features are numbers?', features.every(f => typeof f === 'number'));

    // CONFIRMED: Model expects exactly 10 features (features 1-10 only)
    // Features used by model:
    // 1-4: Eating patterns (campus eatery, prepared meals, eating with friends, eating alone)
    // 5: Nutritional confidence (1-5 scale)
    // 6-10: Employment/Activities (jobs on/off campus, internships on/off campus, undergrad research)
    console.log('✅ [FRONTEND DEBUG] Using first 10 features for model:', features.slice(0, 10));
    console.log('📝 [FRONTEND DEBUG] Excluded features (11-23 - social/support data):', features.slice(10));
    
    // Return only the first 10 features that the model expects
    return features.slice(0, 10);
  };

  // Interpret SageMaker's raw prediction value
  const interpretSageMakerPrediction = (rawValue: number): {
    rawValue: number;
    sigmoidProbability: number;
    normalizedProbability: number;
    suggestedInterpretation: string;
    note: string;
    riskCategory: string;
    recommendedAction: string;
  } => {
    // The model returns a raw logit value (e.g., 3.698)
    // Higher positive values = higher probability of student attrition
    
    // Option 1: If it's already a sigmoid output (0-1), use directly
    if (rawValue >= 0 && rawValue <= 1) {
      return {
        rawValue: rawValue,
        sigmoidProbability: rawValue * 100,
        normalizedProbability: rawValue * 100,
        suggestedInterpretation: rawValue > 0.7 ? 'Higher Risk' : rawValue > 0.4 ? 'Moderate Risk' : 'Lower Risk',
        note: 'Direct probability from sigmoid output',
        riskCategory: rawValue > 0.7 ? 'Critical' : rawValue > 0.4 ? 'Moderate' : 'Low',
        recommendedAction: rawValue > 0.7 ? 'Immediate intervention needed' : rawValue > 0.4 ? 'Monitor closely' : 'Continue standard support'
      };
    }
    
    // Option 2: Raw logit from neural network - apply sigmoid function
    const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
    const sigmoidProb = sigmoid(rawValue);
    
    // Logit interpretation for student attrition:
    // < -2: Very low risk (< 12% attrition probability)
    // -2 to 0: Low risk (12-50% attrition probability)  
    // 0 to 2: Moderate risk (50-88% attrition probability)
    // > 2: High risk (> 88% attrition probability)
    
    let riskCategory: string;
    let recommendedAction: string;
    let interpretation: string;
    
    if (rawValue < -1) {
      riskCategory = 'Low Risk';
      interpretation = 'Low Risk';
      recommendedAction = 'Continue standard academic support';
    } else if (rawValue < 1) {
      riskCategory = 'Moderate Risk';
      interpretation = 'Moderate Risk';
      recommendedAction = 'Increase engagement and monitoring';
    } else if (rawValue < 2.5) {
      riskCategory = 'High Risk';
      interpretation = 'High Risk';
      recommendedAction = 'Immediate intervention and support services';
    } else {
      riskCategory = 'Critical Risk';
      interpretation = 'Critical Risk';
      recommendedAction = 'Urgent comprehensive intervention required';
    }
    
    return {
      rawValue: rawValue,
      sigmoidProbability: sigmoidProb * 100,
      normalizedProbability: Math.max(0, Math.min(100, (rawValue + 3) / 6 * 100)), // Scale -3 to 3 → 0-100%
      suggestedInterpretation: interpretation,
      note: `Logit: ${rawValue.toFixed(3)} → Sigmoid: ${(sigmoidProb * 100).toFixed(1)}% attrition probability`,
      riskCategory: riskCategory,
      recommendedAction: recommendedAction
    };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log('🔍 [FRONTEND DEBUG] ============ FORM SUBMISSION START ============');
    
    if (!formData.rowId.trim()) {
      console.log('🔍 [FRONTEND DEBUG] ❌ Missing RowID, stopping submission');
      alert('Please enter a Student RowID');
      return;
    }
    
    // Echo form data to console as JSON
    console.log('🔍 [FRONTEND DEBUG] Original form data:', JSON.stringify(formData, null, 2));
    
    // Transform to model features and log
    const modelFeatures = transformToModelFeatures(formData);
    console.log('🔍 [FRONTEND DEBUG] Transformed model features:', modelFeatures);
    console.log('🔍 [FRONTEND DEBUG] Features array length:', modelFeatures.length);
    
    // Prepare payload for API
    const apiPayload = { features: modelFeatures };
    console.log('🔍 [FRONTEND DEBUG] API payload:', JSON.stringify(apiPayload, null, 2));
    
    // Call the prediction API
    try {
      console.log('🔍 [FRONTEND DEBUG] ============ CALLING API ============');
      console.log('🔍 [FRONTEND DEBUG] Making fetch request to /api/predict...');
      
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(apiPayload)
      });
      
      console.log('🔍 [FRONTEND DEBUG] Response status:', response.status);
      console.log('🔍 [FRONTEND DEBUG] Response ok:', response.ok);
      
      const prediction = await response.json();
      console.log('🔍 [FRONTEND DEBUG] ============ API RESPONSE ============');
      console.log('🔍 [FRONTEND DEBUG] Full prediction response:', JSON.stringify(prediction, null, 2));
      
      if (!response.ok) {
        console.error('🚨 [FRONTEND ERROR] API returned error status:', response.status);
        console.error('🚨 [FRONTEND ERROR] Error details:', prediction);
        alert(`API Error (${response.status}): ${prediction.error || 'Unknown error'}`);
      } else {
        console.log('🔍 [FRONTEND DEBUG] ✅ API call successful!');
        
        // Store prediction result for dashboard
        setPredictionResult(prediction);
      }
      
    } catch (error) {
      console.error('🚨 [FRONTEND ERROR] ============ FETCH ERROR ============');
      console.error('🚨 [FRONTEND ERROR] Network/fetch error:', error);
      alert('Network error: Failed to reach prediction API. Check console for details.');
    }
    
    console.log('🔍 [FRONTEND DEBUG] ============ SHOWING DASHBOARD ============');
    setShowDashboard(true);
  };

  const handleInputChange = (field: keyof FormData, value: number | boolean | string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  if (showDashboard) {
    const attritionProbability = calculateAttritionProbability(formData);
    
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
            
            {/* Attrition Risk Alert */}
            <div className={`p-6 rounded-lg mb-8 ${attritionProbability > 70 ? 'bg-red-100 border border-primary-red' : attritionProbability > 40 ? 'bg-yellow-100 border border-yellow-500' : 'bg-green-100 border border-green-500'}`}>
              <h2 className="text-2xl font-regular mb-2 text-black">
                Attrition Risk: {attritionProbability}%
              </h2>
              <p className="font-light text-black">
                {attritionProbability > 70 ? 'High Risk - Immediate intervention recommended' : 
                 attritionProbability > 40 ? 'Moderate Risk - Additional support suggested' : 
                 'Low Risk - Continue current support'}
              </p>
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
                <h2 className="text-3xl font-regular text-blue-700 mb-4">🤖 AI Model Prediction</h2>
                
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
                                <span className="font-light text-gray-600">Raw Prediction:</span>
                                <span className="font-regular text-black">{rawValue.toFixed(4)}</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="font-light text-gray-600">Sigmoid Probability:</span>
                                <span className="font-regular text-black">{interpretation.sigmoidProbability.toFixed(1)}%</span>
                              </div>
                              <div className="flex justify-between">
                                <span className="font-light text-gray-600">Risk Assessment:</span>
                                <span className={`font-regular ${
                                  interpretation.riskCategory === 'Critical Risk' ? 'text-red-700' : 
                                  interpretation.riskCategory === 'High Risk' ? 'text-red-600' : 
                                  interpretation.riskCategory === 'Moderate Risk' ? 'text-yellow-600' : 
                                  'text-green-600'
                                }`}>
                                  {interpretation.riskCategory}
                                </span>
                              </div>
                              <div className="pt-3 border-t border-gray-200">
                                <p className="font-light text-gray-600 text-sm mb-2">Recommended Action:</p>
                                <p className="font-regular text-black text-sm bg-blue-50 p-2 rounded">
                                  {interpretation.recommendedAction}
                                </p>
                              </div>
                              <div className="pt-2">
                                <p className="font-light text-gray-500 text-xs">
                                  {interpretation.note}
                                </p>
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
                              <span className="font-regular text-black">{predictionResult.metadata.featuresCount || 'N/A'}/23</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="font-light text-gray-600">Model Type:</span>
                              <span className="font-regular text-black capitalize">{predictionResult.metadata.modelType?.replace('_', ' ') || 'Unknown'}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="font-light text-gray-600">Status:</span>
                              <span className="font-regular text-green-600">✅ Success</span>
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
                    🔧 Raw API Response (for debugging)
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

