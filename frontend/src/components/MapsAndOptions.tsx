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
import { useEffect, useState } from "react";
import { PlaceOfInterest_ } from "./map_utils/types.tsx";

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

    function convertToPlaceOfInterest(obj: {
        [key: string]: string[];
    }): PlaceOfInterest_ {
        // Define valid keys for PlaceOfInterest_
        const validKeys: (keyof PlaceOfInterest_)[] = [
            "education",
            "family",
            "work",
            "hobbies",
            "lifestyle",
        ];

        // Create a new object with the valid keys
        console.log(obj);
        const result: Partial<PlaceOfInterest_> = {};

        for (const key of validKeys) {
            result[key] = obj[key] || [];
        }

        return result as PlaceOfInterest_;
    }

    const [placesOfInterest, setPlacesofInterest] = useState<PlaceOfInterest_>({
        work: [],
        education: [],
        family: [],
        hobbies: [],
        lifestyle: [],
    });

    useEffect(() => {
        console.log("Housing Facts ", housingFacts);
        setPlacesofInterest(convertToPlaceOfInterest(housingFacts));
    }, [housingFacts]);
    return (
        <div className="mapsAndOptions">
            <Maps placesOfInterest={placesOfInterest} />
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
