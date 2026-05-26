import { useState, useEffect, useCallback } from 'react'
import { marketService } from '../services/marketService'

export const useMarketPrices = (state = 'Maharashtra', commodity = 'Tomato', enabled = false) => {
  const [prices, setPrices] = useState([])
  const [source, setSource] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)

  const fetchPrices = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const data = await marketService.getMarketPrices(state, commodity)
      setPrices(data.data)
      setSource(data.source)
      setLastUpdated(data.lastUpdated)
    } catch (err) {
      setError(err.message)
      setPrices([])
    } finally {
      setLoading(false)
    }
  }, [state, commodity])

  useEffect(() => {
    if (!enabled) return
    fetchPrices()
  }, [enabled, fetchPrices])

  return { prices, loading, error, lastUpdated, source, refetch: fetchPrices }
}