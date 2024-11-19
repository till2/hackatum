import "./Blog.css";
import "./components/Accordion.css";
import Template from "./Template";
import BlogEntry from "./components/BlogEntry";

function Blog() {
    return (
        <Template>
            <div className="blogContainer">
                <div className="blog">
                    <h1>Blog</h1>
                    <li>
                        <ul>
                            <BlogEntry title="Test">
                                <div>
                                    <img
                                        src="http://127.0.0.1:8000/demo"
                                        alt="Cat Image"
                                    />
                                </div>
                                <p>
                                    lorem ipsum dolor sit amet, consectetur
                                    adipiscing elit, sed do eiusmod tempor
                                    incididunt ut labore et dolore magna aliqua.
                                    Ut enim ad minim veniam, quis nostrud
                                    exercitation ullamco laboris nisi ut aliquip
                                    ex ea commodo consequat. Duis aute irure
                                    dolor in reprehenderit in voluptate velit
                                    esse cillum dolore eu fugiat nulla pariatur.
                                    Excepteur sint occaecat cupidatat non
                                    proident, sunt in culpa qui officia deserunt
                                    mollit anim id est laborum.
                                </p>
                                <p>
                                    lorem ipsum dolor sit amet, consectetur
                                    adipiscing elit, sed do eiusmod tempor
                                    incididunt ut labore et dolore magna aliqua.
                                    Ut enim ad minim veniam, quis nostrud
                                    exercitation ullamco laboris nisi ut aliquip
                                    ex ea commodo consequat. Duis aute irure
                                    dolor in reprehenderit in voluptate velit
                                    esse cillum dolore eu fugiat nulla pariatur.
                                    Excepteur sint occaecat cupidatat non
                                    proident, sunt in culpa qui officia deserunt
                                    mollit anim id est laborum.
                                </p>
                            </BlogEntry>
                        </ul>
                    </li>
                </div>
            </div>
        </Template>
    );
}

export default Blog;
