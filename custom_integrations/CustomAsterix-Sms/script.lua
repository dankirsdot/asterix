prev_health = 3
prev_lives = 3
prev_bones = 0
prev_item = 0
prev_key = 0
prev_score = 0
prev_time = 0

boss_killed = 0

function asterix_reward ()
    local health_reward = (data.health - prev_health) * 1
    local lives_reward = (data.lives - prev_lives) * 10
    local bones_reward = (data.bones - prev_bones) * 1
    local item_reward = (data.item - prev_item) * 5
    local key_reward = (data.key - prev_key) * 5
    local score_reward = (data.score - prev_score) * 1
    local time_reward = -1/50 

    local level_reward = 0
    if score_reward == 30 then
        boss_killed = 1
        -- level_reward = data.time / 10
    end

    prev_health = data.health
    prev_lives = data.lives
    prev_bones = data.bones
    prev_item = data.item
    prev_key = data.key
    prev_score = data.score
    prev_time = data.time

    local true_reward = score_reward + time_reward + level_reward
    local shaping_reward = health_reward + lives_reward + bones_reward + item_reward + key_reward
    return true_reward + shaping_reward 
end


function asterix_done()
    -- this condition is incorrect because the game ends one life early
    -- but it is the most reliable one we were able to identify
    if data.lives == 0 then
        return true
    end

    -- you can complete the level only if you win a boss fight
    if boss_killed == 1 then
        return true
    end

    return false
end
