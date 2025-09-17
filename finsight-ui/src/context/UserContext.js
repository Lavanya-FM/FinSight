import React, { createContext, useContext, useState, useEffect } from 'react';
import { supabase } from '../supabase';

const UserContext = createContext();

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loading, setLoading] = useState(true);

  const checkSession = async () => {
    console.log('Checking session...');
    try {
      setLoading(true);
      const { data: { session }, error } = await supabase.auth.getSession();
      if (error) throw error;
      if (session?.user) {
        setUser(session.user);
        setIsLoggedIn(true);
        const { data, error: userError } = await supabase
          .from('users')
          .select('role')
          .eq('user_id', session.user.id)
          .single();
        if (userError) throw userError;
        setUser((prev) => ({ ...prev, role: data.role }));
      } else {
        setIsLoggedIn(false);
        setUser(null);
      }
    } catch (err) {
      console.error('Session check failed:', err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async (callback) => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      setUser(null);
      setIsLoggedIn(false);
      console.log('Logged out successfully');
      if (callback) callback();
    } catch (err) {
      console.error('Logout failed:', err.message);
    }
  };

  useEffect(() => {
    checkSession();
    const { data: authListener } = supabase.auth.onAuthStateChange((event, session) => {
      checkSession();
    });
    return () => authListener.subscription.unsubscribe();
  }, []);

  return (
    <UserContext.Provider value={{ user, isLoggedIn, loading, handleLogout }}>
      {children}
    </UserContext.Provider>
  );
};

export const useUser = () => useContext(UserContext);