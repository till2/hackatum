import React, {FunctionComponent} from 'react';

import {FloorplanIcon} from '../../../data/icons/floor-plan-icon';
import {BathroomIcon} from '../../../data/icons/bathroom-icon';
import {BedroomIcon} from '../../../data/icons/bedroom-icon';

import {ListingDetails} from './types';

import './listing_details.css';

interface Props {
  details: ListingDetails;
}

const getFormattedCurrency = (priceString: string) => {
    const price = parseFloat(priceString.replace('$', ''));
    return formatCurrency(price);
  };
  
  const formatCurrency = (price: number) => {
    const formatter = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    });
  
    return formatter.format(price);
  };

export const RealEstateListingDetails: FunctionComponent<Props> = ({
  details
}) => {
  const {
    property_address,
    property_price,
    listing_title,
    property_bedrooms,
    property_bathrooms,
    property_square_feet,
    listing_description
  } = details;

  return (
    <div className="details-container">
      <div className="listing-content">
        <h2>{listing_title}</h2>
        <p>{property_address}</p>
        <div className="details">
          <div className="detail_item">
            <FloorplanIcon /> {property_square_feet.replace('sq ft', 'ft²')}
          </div>
          <div className="detail_item">
            <BathroomIcon /> {property_bathrooms}
          </div>
          <div className="detail_item">
            <BedroomIcon /> {property_bedrooms}
          </div>
        </div>

        <p className="description">{listing_description}</p>

        <p className="price">{getFormattedCurrency(property_price)}</p>
      </div>
    </div>
  );
};
