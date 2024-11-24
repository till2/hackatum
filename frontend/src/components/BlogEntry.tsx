import React from "react";
import "./BlogEntry.css";

interface BlogEntryProps {
    title: string;
    children: React.ReactNode;
}

const BlogEntry: React.FC<BlogEntryProps> = ({ title, children }) => {
    return (
        <div className="blogEntry">
            <h2 className="blogTitle">{title}</h2>
            <div className="blogContent">{children}</div>
        </div>
    );
};

export default BlogEntry;
