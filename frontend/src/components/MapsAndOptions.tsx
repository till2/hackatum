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
import ChatBot from 'react-chatbotify'
import "./chatbot.css"

const MapsAndOptions = ({
    setInputText,
    lifestyles,
    emojis,
    factCategories,
    dictFacts,
    housingFacts,
    startLocation
}: {
    setInputText: React.Dispatch<React.SetStateAction<string>>;
    lifestyles: DictLifestyle;
    emojis: DictEmojis;
    factCategories: DictFactCategories;
    dictFacts: DictFacts;
    housingFacts: DictHousingFacts;
    startLocation: google.maps.LatLngLiteral;
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


    const flow = {
        "start": {
          "message": "Hello! \nI am TumTum, your friendly personal assistant. Would you like to do some brainstorming to make sure your next home is really perfect? \nOr do you need help with understanding some obscure admninistrative detail? The German bureaucracy can be a bit of a maze, but I'm here to help you navigate it!",
        }
      }

    useEffect(() => {
        console.log("Housing Facts ", housingFacts);
        setPlacesofInterest(convertToPlaceOfInterest(housingFacts));
    }, [housingFacts]);

    const input = document.getElementsByTagName("input")[0];

// focus on the input element
    input.focus();

    // add event listeners to the input element
    input.addEventListener('keypress', (event) => {
    console.log("You have pressed key: ", event.key);
    });

    input.addEventListener('keydown', (event) => {
    console.log(`key: ${event.key} has been pressed down`);
    });

    input.addEventListener('keyup', (event) => {
    console.log(`key: ${event.key} has been released`);
    });

    console.log("ddd")
    // dispatch keyboard events
    input.dispatchEvent(new KeyboardEvent('keydown',  {'key':'e'}));
    return (
        <div className="mapsAndOptions">
            <Maps placesOfInterest={placesOfInterest} startLocation={startLocation}/>
            <Options
                setInputText={setInputText}
                lifestyles={lifestyles}
                emojis={emojis}
                factCategories={factCategories}
                dictFacts={dictFacts}
                housingFacts={housingFacts}
            />
        <ChatBot flow={flow}/>
        </div>
        
    );
};
export default MapsAndOptions;
