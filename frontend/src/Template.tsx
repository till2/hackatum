import "./Template.css";
import "./components/Accordion.css";
import Footer from "./components/Footer";
import Header from "./components/Header";

function Template({ children }: { children: JSX.Element }) {
    return (
        <div className="template">
            <Header />
            {children}
            <Footer />
        </div>
    );
}

export default Template;
