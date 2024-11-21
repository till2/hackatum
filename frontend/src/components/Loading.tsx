import React from "react";
import "./Loading.css";
import dogImage from "../../assets/dog.png";

const Loading: React.FC = () => {
    return (
        <div className="loading-container">
            <img src={dogImage} alt="Rotating Dog" className="rotating-dog" />
        </div>
    );
};

export default Loading; 