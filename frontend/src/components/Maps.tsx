import {
    APIProvider,
    Map,
    AdvancedMarker,
    useMapsLibrary,
    useMap,
    InfoWindow,
    useAdvancedMarkerRef,
} from "@vis.gl/react-google-maps";

import { Polyline } from "./drawings/polyline";

import React, { useEffect, useState } from "react";

import frontImage from "../../data/images/front.jpg";
import bedroomImage from "../../data/images/bedroom.jpg";
import backImage from "../../data/images/back.jpg";
import {
    RealEstateListing,
    PlaceOfInterest,
    PlaceOfInterest_,
} from "./map_utils/types";
import { CustomMarker } from "./map_utils/custom_marker";
import { PoiMarker } from "./map_utils/poi_marker";

import "./Maps.css";

const Maps = ({ placesOfInterest }: { placesOfInterest: PlaceOfInterest_ }) => {
    return (
        <div className="custom-marker">
            <APIProvider apiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY}>
                <MarkerHandler placesOfInterest={placesOfInterest} />
            </APIProvider>
        </div>
    );
};

// var select: google.maps.LatLngLiteral | null = null;

export async function loadRealEstateListing(): Promise<RealEstateListing[]> {
    // call backend, get results in current location.
    const url = new URL("../../data/real-estate-listing.json", import.meta.url);

    const listings = (await fetch(url).then((res) =>
        res.json(),
    )) as RealEstateListing[];

    listings.forEach(
        (listing) => (listing.images = [frontImage, bedroomImage, backImage]),
    );

    return listings;
}

const MarkerHandler = ({
    placesOfInterest,
}: {
    placesOfInterest: PlaceOfInterest_;
}) => {
    const [select, setSelect] = useState<RealEstateListing | null>(null);
    const [selectPOI, setSelectPOI] = useState<PlaceOfInterest | null>(null);
    const [hover, setHover] = useState<RealEstateListing | null>(null);
    const [realEstateListings, setRealEstateListing] = useState<
        RealEstateListing[]
    >([]);
    const [placeOfInterest_, setPlaceOfInterest_] = useState<
        PlaceOfInterest_[]
    >([]);
    const [placeOfInterest, setPlaceOfInterest] = useState<PlaceOfInterest[]>(
        [],
    );

    const map = useMap();
    const places = useMapsLibrary("places");

    useEffect(() => {
        loadRealEstateListing().then((data) => {
            setRealEstateListing(data);
        });
    }, []);

    useEffect(() => {
        setPlaceOfInterest_([placesOfInterest]);
    }, [placesOfInterest]);

    useEffect(() => {
        if (!places || !map) return;
        FindPlaces(places, placeOfInterest_[0], setPlaceOfInterest);
    }, [placeOfInterest_, placesOfInterest, places, map]);

    useEffect(() => {}, [select]);
    return (
        <>
            {realEstateListings.length != 0 && (
                <Map
                    style={{
                        width: "100%",
                        height: "calc(100% - 15px)",
                        borderRadius: "8px",
                        overflow: "hidden",
                        marginTop: "5px",
                    }}
                    defaultCenter={{
                        lat: realEstateListings[0].details.latitude,
                        lng: realEstateListings[0].details.longitude,
                    }}
                    defaultZoom={15}
                    gestureHandling={"greedy"}
                    mapId="DEMO_MAP_ID"
                    disableDefaultUI={true}
                    onClick={() => setSelect(null)}
                > 
                {!select && (
                    <DrawLines origin={hover} targets={placeOfInterest} />)}
                </Map>
            )}
            {/* <PoiMarkers pois={locations} setSelect={setSelect}/> */}
            <DisplayRealEstateMarkers
                listings={realEstateListings}
                select={select}
                setSelect={setSelect}
                setHover={setHover}
            />
           <DisplayPOI pois={placeOfInterest} select={select} selectPOI={selectPOI} setSelectPOI={setSelectPOI}/>

        </>
    );
};

const DrawLines = (props: {
    origin: RealEstateListing | null;
    targets: PlaceOfInterest[];
}) => {
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
                <MakeLines
                    center={{
                        lat: props.origin.details.latitude,
                        lng: props.origin.details.longitude,
                    }}
                    ends={props.targets.map((target) => ({
                        lat: target.location.lat,
                        lng: target.location.lng,
                    }))}
                />
            ) : null}
        </>
    );
};

// const setCenter = (center: google.maps.LatLngLiteral) => {
//     console.log(center)
//     select = center;
// }
// @ts-ignore
async function FindPlaces(
    lib,
    query: PlaceOfInterest_,
    seter: React.Dispatch<React.SetStateAction<PlaceOfInterest[]>>,
) {
    query.education.forEach((q) => {
        QueryPlaces(lib, "education", q).then((places) => {
            if (places)
                seter((old_state) => {
                    return [...old_state, ...places];
                });
        });
    });
    query.work.forEach((q) => {
        QueryPlaces(lib, "work", q).then((places) => {
            if (places)
                seter((old_state) => {
                    return [...old_state, ...places];
                });
        });
    });
    query.lifestyle.forEach((q) => {
        QueryPlaces(lib, "lifestyle", q).then((places) => {
            if (places)
                seter((old_state) => {
                    return [...old_state, ...places];
                });
        });
    });
    query.family.forEach((q) => {
        QueryPlaces(lib, "family", q).then((places) => {
            if (places)
                seter((old_state) => {
                    return [...old_state, ...places];
                });
        });
    });

    query.hobbies.forEach((q) => {
        QueryPlaces(lib, "hobbies", q).then((places) => {
            if (places)
                seter((old_state) => {
                    return [...old_state, ...places];
                });
        });
    });
}

//@ts-ignore
async function QueryPlaces(lib, key, value) {
    const request = {
        textQuery: value,
        fields: ["displayName", "location", "formattedAddress"],
        locationBias: { lat: 48.132379, lng: 11.57 },
        language: "en-US",
        maxResultCount: 2,
        region: "de",
        useStrictTypeFiltering: false,
    };

    const { places } = await lib.Place.searchByText(request);
    if (places.length) {
        // Loop through and get all the results.
        const pois: PlaceOfInterest[] = [];
        //@ts-ignore
        places.forEach((place) => {
            // console.log(place.location.lat());
            // console.log(place.location.lng());
            // console.log(place.displayName);
            pois.push({
                displayName: place.displayName,
                formattedAddress: place.formattedAddress,
                key: key,
                value: value,
                location: {
                    lat: place.location.lat(),
                    lng: place.location.lng(),
                },
            });
        });

        return pois;
    } else {
        console.log("No results");
        return null;
    }
}

// }
const DisplayRealEstateMarkers = (props: {
    listings: RealEstateListing[];
    select: RealEstateListing | null;
    setSelect: React.Dispatch<React.SetStateAction<RealEstateListing | null>>;
    setHover: React.Dispatch<React.SetStateAction<RealEstateListing | null>>;
}) => {
    useEffect(() => {}, [props.select]);
    return (
        <>
            {props.listings.map((listing: RealEstateListing) =>
                !props.select || props.select == listing ? (
                    <CustomMarker
                        key={listing.uuid}
                        realEstateListing={listing}
                        select={props.select}
                        setSelect={props.setSelect}
                        setHover={props.setHover}
                    />
                ) : null,
            )}
        </>
    );
};

// }
const DisplayPOI = (props: {pois: PlaceOfInterest[], select: RealEstateListing | null, selectPOI: PlaceOfInterest | null, setSelectPOI: React.Dispatch<React.SetStateAction<PlaceOfInterest | null>>}) => {
  return (
      <>
      {props.pois.map( (poi: PlaceOfInterest, idx: number) =>
        !props.select ? (
            <PoiMarker key={idx} poi={poi} select={props.selectPOI} setSelect={props.setSelectPOI}/>
        ) : null)}
      </>
    );
  };


const MakeLines = ({
    center,
    ends,
}: {
    center: google.maps.LatLngLiteral;
    ends: google.maps.LatLngLiteral[];
}) => {
    return (
        <>
            {ends.map((end, index) => (
                <Polyline
                    key={index}
                    strokeWeight={3}
                    strokeColor={"#FFA500"}
                    path={[center, end]}
                />
            ))}
        </>
    );
};

// const root = createRoot(document.getElementById('app'));
// root.render(<Maps />);

export default Maps;
