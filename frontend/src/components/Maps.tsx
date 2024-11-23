import {
    APIProvider,
    Map,
    AdvancedMarker,
    MapCameraChangedEvent,
    Pin,
    Marker,
    ControlPosition,
    MapControl,
    useMapsLibrary,
    useMap,

  } from '@vis.gl/react-google-maps';

import {Polyline} from './drawings/polyline';

import React, { useEffect, useState } from 'react';

import frontImage from '../../data/images/front.jpg';
import bedroomImage from '../../data/images/bedroom.jpg';
import backImage from '../../data/images/back.jpg';
import {RealEstateListing, PlaceOfInterest} from './map_utils/types';
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


export async function loadPlaceOfInterest(): Promise<PlaceOfInterest[]> {

    // call backend, get results in current location.
    const url = new URL('../../data/place-of-interest.json', import.meta.url);

    const places_of_interest = (await fetch(url).then(res =>
      res.json()
    )) as PlaceOfInterest[];

    return places_of_interest;
  }

const MarkerHandler = () => {
    const [select, setSelect] = useState<RealEstateListing | null>(null);
    const [realEstateListings, setRealEstateListing] = useState<RealEstateListing[]>([]);
    const [placeOfInterest, setPlaceOfInterest] = useState<PlaceOfInterest[]>([]);


    const map = useMap();
    const places = useMapsLibrary("places");
    
    useEffect(() => {
      if (!places || !map) return;
      FindPlaces(places);
    }, [places, map]);
  
    useEffect(() => {
        loadRealEstateListing().then(data => {
          setRealEstateListing(data);
        });
      }, []);
    
    useEffect(() => {
      loadPlaceOfInterest().then(data => {
          setPlaceOfInterest(data);
        });
      }, []);
    
    useEffect(() => {}, [select]);
    return (
        <>
        {realEstateListings.length != 0 &&
        <Map
            style={{ width: "100%", height: "calc(100% - 15px)", borderRadius: "8px", overflow: "hidden", marginTop: "5px" }}
            defaultCenter={{ lat: realEstateListings[0].details.latitude, lng: realEstateListings[0].details.longitude }}
            defaultZoom={15}
            gestureHandling={"greedy"}
            mapId='DEMO_MAP_ID'
            disableDefaultUI={true}
            onClick={() => setSelect(null)}
            >

                <DrawLines origin={select} targets={placeOfInterest[0]} />

            </Map>}
        {/* <PoiMarkers pois={locations} setSelect={setSelect}/> */}
        <DisplayRealEstateMarkers listings={realEstateListings} select={select} setSelect={setSelect}/>
        </>
    )
}

const DrawLines = (props: {origin: RealEstateListing | null, targets: PlaceOfInterest}) => {
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
            {props.origin !== null ? (
                <MakeLines center={{lat: props.origin.details.latitude, lng: props.origin.details.longitude}} ends={props.targets.map(target => ({lat: target.details.latitude, lng: target.details.longitude}))} />
            ) : null}
      </>
    );
  };


// const setCenter = (center: google.maps.LatLngLiteral) => {
//     console.log(center)
//     select = center;
// }

const FindPlaces = (places) => {


  const request = {
      textQuery: "Tacos in Mount`ain View",
      fields: ["displayName", "location", "businessStatus"],
      includedType: "restaurant",
      locationBias: { lat: 37.4161493, lng: -122.0812166 },
      isOpenNow: true,
      language: "en-US",
      maxResultCount: 8,
      minRating: 3.2,
      region: "us",
      useStrictTypeFiltering: false,
  };
  console.log(places)
  const test = places.Place.searchByText(request);
  test.then((response) => {console.log(response)});
  
  return (
    <>
    </>
  )
}


// }
const DisplayRealEstateMarkers = (props: {listings: RealEstateListing[], select: RealEstateListing | null, setSelect: React.Dispatch<React.SetStateAction<RealEstateListing | null>>}) => {
    
  useEffect(() => {}, [props.select]);
  return (
      <>
      
        {props.listings.map( (listing: RealEstateListing) => (
            (!props.select) || (props.select == listing) ? (
              <CustomMarker key={listing.uuid} realEstateListing={listing} select={props.select} setSelect={props.setSelect}/>
            ) : null
        ))}
      </>
    );
  };


const MakeLines = ({ center, ends }: { center: google.maps.LatLngLiteral, ends: google.maps.LatLngLiteral[] }) => {
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
