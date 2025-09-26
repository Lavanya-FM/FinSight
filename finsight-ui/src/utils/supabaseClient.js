// utils/supabaseClient.js - Singleton Supabase client to avoid multiple instances

import { createClient } from '@supabase/supabase-js';

let supabaseClient = null;

const getSupabase = () => {
  if (!supabaseClient) {
    const supabaseUrl = process.env.REACT_APP_SUPABASE_URL;
    const supabaseKey = process.env.REACT_APP_SUPABASE_ANON_KEY;

    if (!supabaseUrl || !supabaseKey) {
      console.error('❌ Missing Supabase environment variables');
      return null;
    }

    supabaseClient = createClient(supabaseUrl, supabaseKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
      },
    });
  }
  return supabaseClient;
};

export default getSupabase();
