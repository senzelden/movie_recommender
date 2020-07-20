-- create tables
CREATE TABLE movies (
	movie_id VARCHAR NOT NULL PRIMARY KEY,
	title VARCHAR(300),
  genre VARCHAR(100)
);

CREATE TABLE links (
  movie_id VARCHAR,
	imdbId VARCHAR(20),
  tmdbId VARCHAR(20),
  FOREIGN KEY (movie_id) REFERENCES movies
);

CREATE TABLE ratings (
  user_id VARCHAR,
  movie_id VARCHAR,
  rating REAL,
  time_stamp VARCHAR(15),
  FOREIGN KEY (movie_id) REFERENCES movies
);

CREATE TABLE tags (
  user_id VARCHAR,
  movie_id VARCHAR NOT NULL,
  tag VARCHAR(100),
  time_stamp VARCHAR(15)
);

-- Inster data into the table
COPY movies FROM '/home/varo/Documents/SPICED_Academy/Week_10/ml-latest-small/movies.csv' DELIMITER ',' CSV HEADER;
COPY ratings FROM '/home/varo/Documents/SPICED_Academy/Week_10/ml-latest-small/ratings.csv' DELIMITER ',' CSV HEADER;
COPY tags FROM '/home/varo/Documents/SPICED_Academy/Week_10/ml-latest-small/tags.csv' DELIMITER ',' CSV HEADER;
COPY links FROM '/home/varo/Documents/SPICED_Academy/Week_10/ml-latest-small/links.csv' DELIMITER ',' CSV HEADER;
