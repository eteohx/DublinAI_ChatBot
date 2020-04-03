ratings = readtable('G:\My Drive\DublinAI\Mini Projects\chatbot\the-movies-dataset\ratings_small.csv');
users = ratings.userId;
movies = ratings.movieId;
rating = ratings.rating;
unique_movies = unique(movies);
unique_users = unique(users);

% make Y matrix (users x movie ratings)
Y = zeros(numel(unique_users),numel(unique_movies));
R = zeros(numel(unique_users),numel(unique_movies));
for i = 1:numel(unique_users)        
    user = unique_users(i);
    i
    for j = 1:numel(unique_movies)
        movie = unique_movies(j);
        r = rating(movies(users==user) == movie);
        if ~isempty(r)
            Y(i,j) = r;
            R(i,j) = 1;            
        end
    end
end



