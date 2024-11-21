import "./Blog.css";
import "./components/Accordion.css";
import Template from "./Template";
import BlogEntry from "./components/BlogEntry";

/*
li: List item for blog entries
ul: Unordered list for blog entries
p: Paragraph for blog entry content
*/

function Blog() {
    return (
        <Template>
            <div className="blogContainer">
                <div className="blog">
                    <h1>Blog</h1>
                    <ul>
                        <li>
                            <BlogEntry title="Test">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/cat.png"
                                        alt="Cat Image"
                                    />
                                    <div className="blogText">
                                        <p>
                                            Lorem ipsum dolor sit amet, consectetur
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
                                            Lorem ipsum dolor sit amet, consectetur
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
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                        <li>
                            <BlogEntry title="Second Test">
                                <div className="blogEntryContent">
                                    <img
                                        src="../../assets/dog.png"
                                        alt="Dog Image"
                                        style={{ width: '50%' }}
                                    />
                                    <div className="blogText">
                                        <p>
                                            Lorem ipsum dolor sit amet, consectetur
                                            adipiscing elit, sed do eiusmod tempor
                                            incididunt ut labore et dolore magna aliqua.
                                            Ut enim ad minim veniam, quis nostrud
                                            exercitation ullamco laboris nisi ut aliquip
                                            ex ea commodo consequat. Duis aute irure
                                            dolor in reprehenderit in voluptate velit
                                            esse cillum dolore eu fugiat nulla pariatur.
                                        </p>
                                    </div>
                                </div>
                            </BlogEntry>
                        </li>
                    </ul>
                </div>
            </div>
        </Template>
    );
}

export default Blog;
