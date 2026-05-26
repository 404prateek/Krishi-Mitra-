import { SCHEMES } from '../data/schemes';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export const schemeService = {
  async getSchemes() {
    try {
      const response = await fetch(`${API_URL}/api/schemes`);
      if (!response.ok) {
        throw new Error(`Scheme fetch failed: ${response.status}`);
      }

      const data = await response.json();
      const schemes = Array.isArray(data?.data) ? data.data : [];
      const hasExpectedShape = schemes.length > 0 && schemes.every((scheme) =>
        typeof scheme?.title === 'string' && typeof scheme?.fullName === 'string'
      );

      if (hasExpectedShape) {
        return schemes;
      }

      return SCHEMES;
    } catch (error) {
      console.warn('schemeService fallback:', error);
      return SCHEMES;
    }
  },
};
