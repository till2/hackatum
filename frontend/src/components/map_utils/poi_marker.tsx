import React, {FunctionComponent, useState} from 'react';
import {AdvancedMarker} from '@vis.gl/react-google-maps';
import classNames from 'classnames';

import {RealEstateListingDetails} from './listing_details';
import {RealEstateGallery} from './gallery';
import {RealEstateIcon} from '../../../data/icons/real-estate-icon';

import {PlaceOfInterest} from './types';

import './custom_marker.css';
import { LiaBasketballBallSolid } from "react-icons/lia";
import { TbBallVolleyball } from "react-icons/tb";
import { PiWine } from "react-icons/pi";
import { PiStudent } from "react-icons/pi";
import { TfiBriefcase } from "react-icons/tfi";


interface Props {
  poi: PlaceOfInterest;
  select: PlaceOfInterest | null
  setSelect: React.Dispatch<React.SetStateAction<PlaceOfInterest | null>>;
}

interface DetailProps {
  address: string;
  displayName: string;
}

export const PoiMarker: FunctionComponent<Props> = ({
  poi,
  select,
  setSelect,
}) => {
  const [hovered, setHovered] = useState(false);
  const position = {
    lat: poi.location.lat,
    lng: poi.location.lng
  };

  const handleClick = (listing: PlaceOfInterest) =>{

    if (listing === select){

      setSelect(null);
    }
    else {
      setSelect(listing);
    }
  }

const POIDetails: FunctionComponent<DetailProps> = ({
    address,
    displayName,
  }) => {
    return (
      <div className="details-container">
        <div className="listing-content">
          <h2>{displayName}</h2>
          <p>{address}</p>
  
          {/* <p className="address">{listing_description}</p> */}
  
        </div>
      </div>
    );
  };
  

  const renderCustomPin = () => {
    return (
      <>
        <div className="custom-pin">
          <button className="close-button">
            <span className="material-symbols-outlined"> close </span>
          </button>

          {/* <div className="image-container"> */}
                <span className="icon" style={{zIndex: 0}}>
                {(poi.key === "lifestyle") && (
                    <PiWine style={{background: "blue", width: "27", height: "27", marginTop: 2, color: "white"}} />
                )};
                {(poi.key === "hobbies") && (
                    <TbBallVolleyball style={{background: "yellow", width: "27", height: "27", marginTop: 2, color: "white"}} />
                    
                )};
                {(poi.key === "education") && (
                    <PiStudent style={{background: "green", width: "27", height: "27", marginTop: 2, color: "white"}} />
                )};
                {(poi.key === "work") && (
                    <TfiBriefcase style={{background: "orange", width: "27", height: "27", marginTop: 2, color: "white"}} />
                )};
                {(poi.key === "family") && (
                    <TfiBriefcase style={{background: "red", width: "27", height: "27", marginTop: 2, color: "white"}} />
                )};
                </span>
          {/* </div> */}

          <POIDetails address={poi.formattedAddress} displayName={poi.displayName}/>
        </div>

        <div className="tip" />
      </>
    );
  };
  const clicked = select === poi;
  return (
    <div className="marker-parent">

      <AdvancedMarker
        position={position}
        title={'Marker for real estate.'}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={classNames('real-estate-marker', {clicked, hovered})}
        onClick={() => handleClick(poi)}>
        {renderCustomPin()}
      </AdvancedMarker>

    </div>
  );
};