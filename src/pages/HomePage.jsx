import Hero from '../components/Hero'
import Features from '../components/Features'
import HowItWorks from '../components/HowItWorks'
import ScanHistory from '../components/ScanHistory'

const HomePage = () => {
  return (
    <div>
      <Hero />
      <ScanHistory />
      <Features />
      <HowItWorks />
    </div>
  )
}

export default HomePage