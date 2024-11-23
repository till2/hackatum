import Maps from "./Maps";
import Options from "./Options";
import "./MapsAndOptions.css";

const MapsAndOptions = ({
    setInputText,
}: {
    setInputText: React.Dispatch<React.SetStateAction<string>>;
}) => {
    return (
        <div className="mapsAndOptions">
            <Maps style={{}} />
            <Options setInputText={setInputText} />
        </div>
    );
};
export default MapsAndOptions;
