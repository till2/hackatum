# MovingTUMorrow 🏡🚀
> Making the search for your forever home more time-efficient and convenient by connecting life planning, housing, and financial data in a shared agentic architecture with a beautiful visualiser.

## Inspiration 💡
Finding a home is often more complex than just picking a place—it's about envisioning your future. Whether it's ensuring proximity to schools for your kids, a gym for your hobbies, or planning a mortgage, the process can feel daunting. We wanted to cut through the noise and create a smarter, simpler, and more personalized way to navigate housing decisions.

## What it does
MovingTUMorrow combines a complex interaction graph of AI agents with an intuitive user interface to simplify housing and financial decisions:

1. **Interactive Life Planner**
   - Helps users simulate different life scenarios and paths
   - Aligns housing needs with future plans through human-in-the-loop feedback
   - Allows modification and selection of proposed scenarios

2. **Smart Housing Matches**
   - Provides tailored recommendations based on:
     - User preferences and budget
     - Proximity to schools, family, sports facilities
     - Cultural and lifestyle activities
     - Match between house features and life planning choices

3. **Interactive Map Visualizer**
   - Displays housing options with detailed information markers
   - Shows distances to all places of interest 
   - Updates dynamically based on life planning choices

4. **Finance Advisor**
   - Connects directly to banks via PSD2
   - Automates financial data processing
   - Provides personalized mortgage options

## How we built it 🛠️
- **AI Core**: LangChain graphs with Google Vertex AI for interactive decision-making
- **Data Integration**: Google Maps API, ThinkImmo Business API, ImmoScout24 API
- **Finance**: PSD2 integration for automated financial processing
- **Frontend**: Interactive map visualizer using Google Maps API

## Challenges we ran into
- **API Reliability**: The provided APIs had limitations we needed to work around
  - ThinkImmo API works only for new housing data polling
  - ImmoScout24 required paid access, leading to mockup development
- **Interface Design**: Creating simple workflows for complex processes

## Accomplishments 🏆
- Built a user-centric platform connecting housing needs with data and financial realities
- Developed a scalable solution that can be easily extended to other cities
- Created a clear, multi-step process that makes house hunting engaging

## What we learned
- Personalization is key - housing isn't one-size-fits-all
- Integrating multiple data sources creates powerful synergies
- Frontend development skills (from a team of ML engineers!) 🤓

## What's next
- **API Integration**: Implementing full API functionality
- **City Expansion**: Adding more German cities and regions
- **Financial Tools**: Adding tax benefits, renovation costs, and savings calculators
- **Gamified Experience**: Adding progress badges and milestones

<em> This project is our contribution to improving the housing crisis one small step at a time. </em>