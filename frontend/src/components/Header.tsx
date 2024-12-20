import Logo from "./Logo";
import MovingTUMorrowBanner from "../../assets/movingtumorrow_banner_cropped.png";
import "./Header.css";
import { Link, useLocation } from "react-router-dom";

const Header = ({disableLogo}: {disableLogo: boolean}) => {
    const location = useLocation();
    const link = location.pathname === "/blog" ? "/" : "/blog";
    const linkText = location.pathname === "/blog" ? "Home" : "Blog";
    return (
        <div className="header">
            {!disableLogo ? (<Link to="/">
                <Logo src={MovingTUMorrowBanner} width="200px" />
            </Link> ) : <div></div>}
            
            <Link style={{ margin: "20px" }} to={link}>{linkText}</Link>
        </div>
    );
};
export default Header;
