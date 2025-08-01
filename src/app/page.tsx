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
  seekingAdvice: number;
  
  // Additional Social & Support factors
  seekAssistanceRHC: string; // Agree/Neutral/Disagree
  seekAssistanceRA: string; // Agree/Neutral/Disagree
  useSharedLivingAgreement: string; // Agree/Neutral/Disagree
  seekAdviceFamily: string; // Agree/Neutral/Disagree
  initiateOpenCommunication: string; // Agree/Neutral/Disagree
  roommateRelationship: string; // Satisfied/Neutral/Unsatisfied/Other
  suiteApartmentOccupants: number; // 1-5+
}

export default function Home() {
  const [showForm, setShowForm] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
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
    seekingAdvice: 3,
    seekAssistanceRHC: 'Neutral',
    seekAssistanceRA: 'Neutral',
    useSharedLivingAgreement: 'Neutral',
    seekAdviceFamily: 'Neutral',
    initiateOpenCommunication: 'Neutral',
    roommateRelationship: 'Neutral',
    suiteApartmentOccupants: 2,
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
    riskScore += data.seekingAdvice > 7 ? 3 : 0;
    
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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.rowId.trim()) {
      alert('Please enter a Student RowID');
      return;
    }
    
    // Echo form data to console as JSON
    console.log('Form Data Submitted:', JSON.stringify(formData, null, 2));
    
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
                  <p>Frequency Seeking Advice: {formData.seekingAdvice}/10</p>
                  <p>Seeks Assistance from RHC: {formData.seekAssistanceRHC}</p>
                  <p>Seeks Assistance from RA/CA: {formData.seekAssistanceRA}</p>
                  <p>Uses Shared Living Agreement: {formData.useSharedLivingAgreement}</p>
                  <p>Seeks Advice from Family: {formData.seekAdviceFamily}</p>
                  <p>Initiates Open Communication: {formData.initiateOpenCommunication}</p>
                  <p>Roommate Relationship (30 days): {formData.roommateRelationship}</p>
                  <p>Suite/Apartment Occupants: {formData.suiteApartmentOccupants === 5 ? '5+' : formData.suiteApartmentOccupants}</p>
                </div>
              </div>
            </div>
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
                    <label className="block font-regular text-gray-700 mb-2">How often student seeks advice (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.seekingAdvice}
                      onChange={(e) => handleInputChange('seekingAdvice', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.seekingAdvice}</span>
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
