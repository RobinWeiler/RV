def _toggle_modal(list_button_clicks, is_open):
    """Indicates whether modal should be open or closed based on relevant button clicks.

    Args:
        list_button_clicks (list of ints): List of clicked buttons.
        is_open (bool): Whether or not modal is currently open.

    Returns:
        bool: Whether or not modal should now be open.
    """
    for button in list_button_clicks:
        if button:
            return not is_open
    return is_open
