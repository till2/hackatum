import { APIProvider, Map } from "@vis.gl/react-google-maps";

const Maps = () => {
    return (
        <APIProvider apiKey={import.meta.env.VITE_GOOGLE_MAPS_API_KEY}>
            <Map
                style={{ width: "50vw", height: "100vh" }}
                defaultCenter={{ lat: 22.54992, lng: 0 }}
                defaultZoom={3}
                gestureHandling={"greedy"}
                disableDefaultUI={true}
            />
        </APIProvider>
    );
};

export default Maps;
