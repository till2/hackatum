import React, {FunctionComponent, useState} from 'react';
import {AdvancedMarker} from '@vis.gl/react-google-maps';
import classNames from 'classnames';

import {RealEstateListingDetails} from './listing_details';
import {RealEstateGallery} from './gallery';
import {RealEstateIcon} from '../../../data/icons/real-estate-icon';

import {RealEstateListing} from './types';

import './custom_marker.css';


interface Props {
  realEstateListing: RealEstateListing;
  select: RealEstateListing | null
  setSelect: React.Dispatch<React.SetStateAction<RealEstateListing | null>>;
}

export const CustomMarker: FunctionComponent<Props> = ({
  realEstateListing,
  select,
  setSelect,
}) => {
  const [hovered, setHovered] = useState(false);
  const position = {
    lat: realEstateListing.details.latitude,
    lng: realEstateListing.details.longitude
  };

  const handleClick = (listing: RealEstateListing) =>{

    if (listing === select){

      setSelect(null);
    }
    else {
      setSelect(listing);

    }
  }
  
  const renderCustomPin = () => {
    return (
      <>
        <div className="custom-pin">
          <button className="close-button">
            <span className="material-symbols-outlined"> close </span>
          </button>

          <div className="image-container">
            <RealEstateGallery
              images={realEstateListing.images}
              isExtended={select === realEstateListing}
            />
            <span className="icon" style={{zIndex: 0}}>
              <RealEstateIcon />
            </span>
          </div>

          <RealEstateListingDetails details={realEstateListing.details} />
        </div>

        <div className="tip" />
      </>
    );
  };

  const clicked = select === realEstateListing;
  return (
    <div className="marker-parent">

      <AdvancedMarker
        position={position}
        title={'Marker for real estate.'}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={classNames('real-estate-marker', {clicked, hovered})}
        onClick={() => handleClick(realEstateListing)}>
        {renderCustomPin()}
      </AdvancedMarker>

    </div>
  );
};