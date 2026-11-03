import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import DataCenterGuardian from './DataCenterGuardian.tsx'
import { applyTheme, initialTheme } from './theme'

// Apply the theme before the first React render to avoid a flash.
applyTheme(initialTheme())

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <DataCenterGuardian />
  </StrictMode>,
)
