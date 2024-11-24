import "./Template.css";
import "./components/Accordion.css";
import Footer from "./components/Footer";
import Header from "./components/Header";


function Template({ disableLogo, children }: { disableLogo: boolean, children: JSX.Element }) {
    return (
        <div className="template">
            <Header disableLogo={disableLogo}/>
            {children}
            <Footer />
        </div>
    );
}

export default Template;
