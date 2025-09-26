//src/supabase.js - Singleton Supabase client to avoid multiple instances
// Enhanced for both local development and cloud deployment

import { createClient } from '@supabase/supabase-js';

let supabaseClient = null;
let supabaseErrorLogged = false;

const getSupabase = () => {
  // Return existing client if already created
  if (supabaseClient) {
    return supabaseClient;
  }

  // Get environment variables
  const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
  const supabaseAnonKey = process.env.REACT_APP_SUPABASE_ANON_KEY;

  // Validate configuration
  if (!supabaseUrl || !supabaseAnonKey) {
    if (!supabaseErrorLogged) {
      console.error('Supabase configuration missing:', {
        hasUrl: !!supabaseUrl,
        hasKey: !!supabaseAnonKey,
        environment: process.env.NODE_ENV,
        timestamp: new Date().toISOString()
      });
      console.error('Please check your environment variables: REACT_APP_SUPABASE_URL and REACT_APP_SUPABASE_ANON_KEY');
      supabaseErrorLogged = true;
    }

    // Return null to prevent app crashes
    return null;
  }

  try {
    // Create Supabase client with enhanced configuration
    supabaseClient = createClient(supabaseUrl, supabaseAnonKey, {
      auth: {
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: true,
        flowType: 'pkce'
      },
      realtime: {
        params: {
          eventsPerSecond: 10
        }
      },
      global: {
        headers: {
          'X-Client-Info': 'finsight-ui@1.0.0'
        }
      }
    });

    console.log('Supabase client initialized successfully:', {
      environment: process.env.NODE_ENV,
      hasAuth: !!supabaseClient.auth,
      hasRealtime: !!supabaseClient.realtime
    });

    return supabaseClient;
  } catch (error) {
    console.error('Failed to initialize Supabase client:', error);
    supabaseErrorLogged = true;
    return null;
  }
};

// Initialize Supabase client
export const supabase = getSupabase();

// Helper function to check if Supabase is properly configured
export const isSupabaseConfigured = () => {
  return supabase !== null;
};

// Helper function to get Supabase configuration status
export const getSupabaseStatus = () => {
  return {
    isConfigured: isSupabaseConfigured(),
    hasAuth: supabase?.auth !== undefined,
    hasRealtime: supabase?.realtime !== undefined,
    environment: process.env.NODE_ENV,
    timestamp: new Date().toISOString()
  };
};