import {
    APIProvider,
    Map,
    AdvancedMarker,
    MapCameraChangedEvent,
    Pin,
    Marker,
    ControlPosition,
    MapControl,
    
  } from '@vis.gl/react-google-maps';

import {Polyline} from './drawings/polyline';

import React, { useEffect, useState } from 'react';


import frontImage from '../../data/images/front.jpg';
import bedroomImage from '../../data/images/bedroom.jpg';
import backImage from '../../data/images/back.jpg';
import {RealEstateListing} from './map_utils/types';
import {CustomMarker} from './map_utils/custom_marker';

import './Maps.css';

const Maps = () => {
    return (
        <div className="custom-marker">

        <APIProvider apiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY}> 
            <MarkerHandler/>
        </APIProvider>

        </div>
    );
};

// var select: google.maps.LatLngLiteral | null = null;

export async function loadRealEstateListing(): Promise<RealEstateListing[]> {

    // call backend, get results in current location.
    const url = new URL('../../data/real-estate-listing.json', import.meta.url);
  
    const listings = (await fetch(url).then(res =>
      res.json()
    )) as RealEstateListing[];
    
    listings.forEach(listing => listing.images = [frontImage, bedroomImage, backImage]);

    return listings;
  }

const MarkerHandler = () => {
    const [select, setSelect] = useState<google.maps.LatLngLiteral | null>(null);
    const [realEstateListings, setRealEstateListing] = useState<RealEstateListing[]>([]);
    
    useEffect(() => {
        loadRealEstateListing().then(data => {
          setRealEstateListing(data);
        });
      }, []);
    console.log(realEstateListings)
    return (
        <>
        {realEstateListings.length != 0 &&
        <Map
            style={{ width: "50vw", height: "100vh" }}
            defaultCenter={{ lat: realEstateListings[0].details.latitude, lng: realEstateListings[0].details.longitude }}
            defaultZoom={15}
            gestureHandling={"greedy"}
            mapId='DEMO_MAP_ID'
            disableDefaultUI={true}
            >

                {/* <DrawingExample select={select} setSelect={setSelect}/> */}

            </Map>}
        {/* <PoiMarkers pois={locations} setSelect={setSelect}/> */}
        <DisplayMarkers listings={realEstateListings} />
        </>
    )
}

const DrawingExample = (props: {select: google.maps.LatLngLiteral | null, setSelect: React.Dispatch<React.SetStateAction<google.maps.LatLngLiteral | null>>}) => {
    return (
        <>
            
            {/* <Marker
                position={center}
                draggable
                onDrag={e =>
                    setCenter({lat: e.latLng?.lat() ?? 0, lng: e.latLng?.lng() ?? 0})
                }
                /> */}
            {/* <Polyline
                strokeWeight={3}
                strokeColor={'#00'}
                path={flightPlanCoordinates}
                /> */}
            {props.select !== null ? (
            <MakeLines center={props.select} ends={flightPlanCoordinates} />
            ) : null}
      </>
    );
  };


// const setCenter = (center: google.maps.LatLngLiteral) => {
//     console.log(center)
//     select = center;
// }

const DisplayMarkers = (props: {listings: RealEstateListing[]}) => {
    return (
      <>
        {props.listings.map( (listing: RealEstateListing) => (
          <CustomMarker realEstateListing={listing} />
        ))}
      </>
    );
  };


const MakeLines = ({ center, ends }: { center: google.maps.LatLngLiteral, ends: google.maps.LatLngLiteral[] }) => {
    console.log(center)
    console.log(ends)
    return (
      <>
        {ends.map((end) => (
          <Polyline
            // key={index}
            strokeWeight={3}
            strokeColor={'#00'}
            path={[
              center,
              end,
            ]}
          />
        ))}
      </>
    );
};

// const root = createRoot(document.getElementById('app'));
// root.render(<Maps />);

export default Maps;
