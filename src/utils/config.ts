export const getApiBaseUrl = (): string => {
  // If Vite frontend is accessed via a LAN IP or custom domain,
  // we dynamically use the same host but target the backend port (5000)
  const hostname = typeof window !== 'undefined' ? window.location.hostname : '127.0.0.1';
  return `http://${hostname}:5000`;
};
