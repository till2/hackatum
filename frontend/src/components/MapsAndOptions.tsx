import Maps from "./Maps";
import Options from "./Options";
import "./MapsAndOptions.css";
import { useEffect } from "react";

const MapsAndOptions = ({
    setInputText,
}: {
    setInputText: React.Dispatch<React.SetStateAction<string>>;
}) => {
    useEffect(() => {console.log(setInputText)}, []);
    return (
        <div className="mapsAndOptions">
            <Maps />
            <Options setInputText={setInputText} />
        </div>
    );
};
export default MapsAndOptions;
