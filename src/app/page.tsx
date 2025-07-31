"use client";

import Image from "next/image";
import { useState } from "react";

interface FormData {
  // Student Identifier
  rowId: string;
  studentName: string;
  
  // Mental Health
  mentalHealthRating: number;
  reachOutToCounselor: boolean;
  avoidConfrontation: boolean;
  senseOfBelonging: number;
  roomRequest: boolean;
  
  // Nutritional Health
  eatingHabits: number;
  eatingAlone: number;
  eatingWithFriends: number;
  preparedMealAlone: number;
  
  // Job Information
  jobOnCampus: boolean;
  jobOffCampus: boolean;
  internshipOnCampus: boolean;
  internshipOffCampus: boolean;
  undergradResearch: boolean;
  
  // Sense of belonging
  attendedEvent: boolean;
  
  // Other factors
  roommateConflicts: number;
  seekingAdvice: number;
}

export default function Home() {
  const [showForm, setShowForm] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    rowId: '',
    studentName: '',
    mentalHealthRating: 5,
    reachOutToCounselor: false,
    avoidConfrontation: false,
    senseOfBelonging: 5,
    roomRequest: false,
    eatingHabits: 5,
    eatingAlone: 3,
    eatingWithFriends: 3,
    preparedMealAlone: 3,
    jobOnCampus: false,
    jobOffCampus: false,
    internshipOnCampus: false,
    internshipOffCampus: false,
    undergradResearch: false,
    attendedEvent: false,
    roommateConflicts: 1,
    seekingAdvice: 3,
  });

  const calculateAttritionProbability = (data: FormData): number => {
    let riskScore = 0;
    
    // Mental health factors (higher weight)
    riskScore += (10 - data.mentalHealthRating) * 3;
    riskScore += data.reachOutToCounselor ? -2 : 2;
    riskScore += data.avoidConfrontation ? 3 : 0;
    riskScore += (10 - data.senseOfBelonging) * 2;
    riskScore += data.roomRequest ? 2 : 0;
    
    // Nutritional health
    riskScore += (10 - data.eatingHabits) * 1.5;
    riskScore += data.eatingAlone > 5 ? 2 : 0;
    riskScore += data.eatingWithFriends < 3 ? 2 : 0;
    riskScore += data.preparedMealAlone > 7 ? 1 : 0;
    
    // Job/engagement factors (protective)
    riskScore += data.jobOnCampus ? -3 : 1;
    riskScore += data.jobOffCampus ? -1 : 0;
    riskScore += data.internshipOnCampus ? -4 : 0;
    riskScore += data.internshipOffCampus ? -2 : 0;
    riskScore += data.undergradResearch ? -5 : 0;
    
    // Social engagement
    riskScore += data.attendedEvent ? -2 : 2;
    
    // Conflict and support
    riskScore += data.roommateConflicts * 1.5;
    riskScore += data.seekingAdvice > 7 ? 3 : 0;
    
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
                  <p>Mental Health Rating: {formData.mentalHealthRating}/10</p>
                  <p>Has Reached Out to Counselor: {formData.reachOutToCounselor ? 'Yes' : 'No'}</p>
                  <p>Tends to Avoid Confrontation: {formData.avoidConfrontation ? 'Yes' : 'No'}</p>
                  <p>Sense of Belonging Score: {formData.senseOfBelonging}/10</p>
                  <p>Room Change Requests: {formData.roomRequest ? 'Yes' : 'No'}</p>
                </div>
              </div>

              {/* Nutritional Health */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h3 className="text-xl font-regular text-primary-red mb-4">Nutritional Health Data</h3>
                <div className="space-y-2 font-light text-black">
                  <p>Eating Habits Rating: {formData.eatingHabits}/10</p>
                  <p>Frequency Eating Alone: {formData.eatingAlone}/10</p>
                  <p>Frequency Eating with Friends: {formData.eatingWithFriends}/10</p>
                  <p>Prepared Meals Alone: {formData.preparedMealAlone}/10</p>
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
                  <p>Roommate Conflicts Level: {formData.roommateConflicts}/10</p>
                  <p>Frequency Seeking Advice: {formData.seekingAdvice}/10</p>
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
                    <label className="block font-regular text-gray-700 mb-2">Student's mental health rating (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.mentalHealthRating}
                      onChange={(e) => handleInputChange('mentalHealthRating', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.mentalHealthRating}</span>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">Student's sense of belonging (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.senseOfBelonging}
                      onChange={(e) => handleInputChange('senseOfBelonging', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.senseOfBelonging}</span>
                  </div>
                  
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      checked={formData.reachOutToCounselor}
                      onChange={(e) => handleInputChange('reachOutToCounselor', e.target.checked)}
                      className="mr-3"
                    />
                    <label className="font-light text-gray-700">Student has reached out to counselor</label>
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
                    <label className="block font-regular text-gray-700 mb-2">Student's eating habits rating (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.eatingHabits}
                      onChange={(e) => handleInputChange('eatingHabits', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.eatingHabits}</span>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">How often student eats alone (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.eatingAlone}
                      onChange={(e) => handleInputChange('eatingAlone', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.eatingAlone}</span>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">How often student eats with friends (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.eatingWithFriends}
                      onChange={(e) => handleInputChange('eatingWithFriends', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.eatingWithFriends}</span>
                  </div>
                  
                  <div>
                    <label className="block font-regular text-gray-700 mb-2">How often student prepares meals alone (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.preparedMealAlone}
                      onChange={(e) => handleInputChange('preparedMealAlone', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.preparedMealAlone}</span>
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
                    <label className="block font-regular text-gray-700 mb-2">Student's roommate conflicts level (1-10)</label>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.roommateConflicts}
                      onChange={(e) => handleInputChange('roommateConflicts', parseInt(e.target.value))}
                      className="w-full"
                    />
                    <span className="font-light text-sm text-black">Current: {formData.roommateConflicts}</span>
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
