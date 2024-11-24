import Maps from "./Maps";
import Options from "./Options";
import "./MapsAndOptions.css";
import {
    DictLifestyle,
    DictEmojis,
    DictFactCategories,
    DictFacts,
    DictHousingFacts,
} from "../types.ts";

const MapsAndOptions = ({
    setInputText,
    lifestyles,
    emojis,
    factCategories,
    dictFacts,
    housingFacts,
}: {
    setInputText: React.Dispatch<React.SetStateAction<string>>;
    lifestyles: DictLifestyle;
    emojis: DictEmojis;
    factCategories: DictFactCategories;
    dictFacts: DictFacts;
    housingFacts: DictHousingFacts;
}) => {
    return (
        <div className="mapsAndOptions">
            <Maps />
            <Options
                setInputText={setInputText}
                lifestyles={lifestyles}
                emojis={emojis}
                factCategories={factCategories}
                dictFacts={dictFacts}
                housingFacts={housingFacts}
            />
        </div>
    );
};
export default MapsAndOptions;
