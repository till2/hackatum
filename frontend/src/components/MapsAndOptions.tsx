import Maps from "./Maps";
import Options from "./Options";
import "./MapsAndOptions.css";
import { DictLifestyle } from "../types.ts";

const MapsAndOptions = ({
    setInputText,
    lifestyles,
}: {
    setInputText: React.Dispatch<React.SetStateAction<string>>;
    lifestyles: DictLifestyle;
}) => {
    return (
        <div className="mapsAndOptions">
            <Maps />
            <Options setInputText={setInputText} />
        </div>
    );
};
export default MapsAndOptions;
